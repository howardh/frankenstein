from collections import OrderedDict, defaultdict
import copy
import datetime
import itertools
import os
import time
from typing import Generic, TypeVar, TypedDict, Callable, overload

import gymnasium
import numpy as np
import tabulate
import torch
import torch.nn
import torch.nn.utils

from frankenstein.buffer import HistoryBuffer
from frankenstein.algorithms.trainer import Trainer
from frankenstein.algorithms.utils import to_tensor, get_action_dist_function, FeedforwardModel, RecurrentModel, format_rate
from frankenstein.buffer.vec_history import VecHistoryBuffer


##################################################
# Config


class SACConfig(TypedDict, total=False):
    buffer_size: int
    warmup_steps: int # Number of transitions to collect before training starts
    batch_size: int
    discount: float
    policy_update_frequency: int # Number of gradient steps between policy network updates
    target_update_frequency: int # Number of gradient steps between target network updates
    target_update_rate: float # Polyak averaging rate for target network updates. 1 means hard update. 0 means no update.
    update_frequency: int # Number of transitions between gradient steps
    entropy_coeff: float


DEFAULT_SAC_CONFIG = SACConfig(
    buffer_size = 1_000_000,
    warmup_steps = 5_000,
    batch_size = 256,
    discount = 0.99,
    policy_update_frequency=2,
    target_update_frequency=1,
    target_update_rate=0.005,
    update_frequency=1,
    entropy_coeff = 0.05,
)


##################################################
# Callbacks


class SACCallbacks():
    def on_start(self, l): ...
    def on_end(self, l): ...
    def on_warmup_start(self, l): ...
    def on_warmup_end(self, l): ...
    def on_training_start(self, l): ...
    def on_gather_start(self, l): ...
    def on_gather_end(self, l): ...
    def on_transition(self, l): ...
    def on_episode_end(self, l): ...
    def on_step_start(self, l): ...
    def on_gradients_end(self, l): ...
    def on_epoch_end(self, l): ...


class VerboseLoggingCallbacks(SACCallbacks):
    def __init__(self):
        self._episode_reward = []
        self._episode_true_reward = []
        self._episode_steps = []

        # Total
        self._start_time = time.time()
        self._total_completed_episodes = 0

        # Since last report
        self._num_completed_episodes = 0
        self._last_report_steps = 0

        self._last_report_time = time.time() # When the last report was printed
        self._report_interval = 5.0 # How often to print a report in seconds

    def on_start(self, l):
        num_envs = l['self'].env.num_envs
        self._total_ep_reward = np.zeros(num_envs, dtype=float)
        self._ep_length = np.zeros(num_envs, dtype=int)

    def on_transition(self, l):
        self._total_ep_reward += l['reward']
        self._ep_length += 1

    def on_episode_end(self, l):
        done = l['terminated'] | l['truncated']
        self._episode_reward.extend(self._total_ep_reward[done].tolist())
        self._episode_steps.extend(self._ep_length[done].tolist())
        self._num_completed_episodes += done.sum()
        self._total_completed_episodes += done.sum()

        self._total_ep_reward[done] = 0
        self._ep_length[done] = 0

        # Check for RecordEpisodeStatistics wrapper
        info = l['info']
        if 'episode' in info:
            r_idx = info['episode']['_r']
            r = info['episode']['r'][r_idx]
            self._episode_true_reward.extend(r.tolist())

    def on_step_start(self, l):
        if self._num_completed_episodes == 0:
            return

        time_diff = time.time() - self._last_report_time
        if time_diff > self._report_interval:
            try:
                term_width, _ = os. get_terminal_size()
            except:
                term_width = 80
            print(f'Time: {datetime.datetime.now()}')
            print(f'Completed step {l["step"]} ({format_rate(l["step"], "step", "steps", time.time() - self._start_time)})')
            #if self._num_completed_episodes == 0:
            #    print('No completed episodes')
            #    return
            print(f'Completed episode(s): {self._num_completed_episodes} ({format_rate(self._num_completed_episodes, "episode", "episodes", time_diff)})')

            if len(self._episode_true_reward) == 0:
                data = OrderedDict([
                    ('Episode reward', self._episode_reward),
                    ('Episode steps', self._episode_steps),
                ])
            else:
                data = OrderedDict([
                    ('Episode reward', self._episode_reward),
                    ('Episode true reward', self._episode_true_reward),
                    ('Episode steps', self._episode_steps),
                ])

            if self._num_completed_episodes == 1:
                rows: list = [(k, v[0]) for k,v in data.items()]
                table = tabulate.tabulate(rows)
            else:
                rows: list = [(k, *v, np.mean(v), np.std(v)) for k,v in data.items()]
                headers = ['Metric', *[str(x) for x in range(self._num_completed_episodes)], 'mean', 'std']
                table = tabulate.tabulate(rows, headers=headers)
                if len(table) > term_width and len(rows[0]) > 10+3:
                    # Produce a smaller table
                    rows: list = [(k, *v[:10], '...', np.mean(v), np.std(v)) for k,v in data.items()]
                    headers = ['Metric', *[str(x) for x in range(10)], '...', 'mean', 'std']
                    table = tabulate.tabulate(rows, headers=headers)
            print(table)

            self._last_report_time = time.time()
            self._episode_reward.clear()
            self._episode_true_reward.clear()
            self._episode_steps.clear()
            self._num_completed_episodes = 0

            print()


class WandbLoggingCallbacks(SACCallbacks):
    def __init__(self, project='rl', **kwargs):
        super().__init__()
        try:
            import wandb # type: ignore
            self._run = wandb.init(
                project=project,
                **kwargs,
            )
        except:
            self._run = None
            pass

        self._transition_count = 0
        self._state_values = []
        self._entropy = []
        self._kl = []
        self._total_ep_reward = np.array(0.)
        self._ep_length = np.array(0)

        self._data = {}

    def on_start(self, l):
        if self._run is None:
            return

        num_envs = l['self'].env.num_envs
        self._total_ep_reward = np.zeros(num_envs, dtype=float)
        self._ep_length = np.zeros(num_envs, dtype=int)

        self._mujoco_reward_components = defaultdict(lambda: np.zeros(num_envs))

        # TODO: Update config

    def on_transition(self, l):
        if self._run is None:
            return

        if 'action_dist' in l:
            self._entropy.append(l['action_dist'].entropy().mean().item())
        self._transition_count += l['num_envs']

        info = l['info']
        for k,v in info.items():
            if k.startswith('reward_'):
                self._mujoco_reward_components[k] += v

        self._total_ep_reward += l['reward']
        self._ep_length += 1

    def on_gradients_end(self, l):
        if 'actor_loss' in l:
            self._data['loss/actor'] = l['actor_loss'].item()
            self._data['state_action_value'] = l['q'].mean().item()
        if 'critic_loss' in l:
            self._data['loss/critic'] = l['critic_loss'].item()

    def on_step_start(self, l):
        if self._run is None:
            return

        if len(self._entropy) > 0:
            self._data['entropy'] = np.mean(self._entropy)

        if len(self._data) > 0:
            self._run.log(self._data, step=self._transition_count)

        self._data = {}
        self._state_values = []
        self._entropy = []

    def on_episode_end(self, l):
        if self._run is None:
            return

        done = l['terminated'] | l['truncated']
        self._data['episode reward'] = self._total_ep_reward[done].mean()
        self._data['episode length'] = self._ep_length[done].mean()

        for k,v in self._mujoco_reward_components.items():
            self._data[f'reward components/{k}'] = v[done].mean()
            v[done] = 0

        self._total_ep_reward[done] = 0
        self._ep_length[done] = 0

        info = l['info']
        if 'episode' in info:
            r_idx = info['episode']['_r']
            r = info['episode']['r'][r_idx]
            self._data['true reward'] = np.mean(r)


class ComposeCallbacks(SACCallbacks):
    def __init__(self, callbacks: list[SACCallbacks] = []):
        self._callbacks = callbacks
    def on_start(self, l):
        for c in self._callbacks:
            c.on_start(l)
    def on_end(self, l):
        for c in self._callbacks:
            c.on_end(l)
    def on_warmup_start(self, l):
        for c in self._callbacks:
            c.on_warmup_start(l)
    def on_warmup_end(self, l):
        for c in self._callbacks:
            c.on_warmup_end(l)
    def on_training_start(self, l):
        for c in self._callbacks:
            c.on_training_start(l)
    def on_gather_start(self, l):
        for c in self._callbacks:
            c.on_gather_start(l)
    def on_gather_end(self, l):
        for c in self._callbacks:
            c.on_gather_end(l)
    def on_transition(self, l):
        for c in self._callbacks:
            c.on_transition(l)
    def on_episode_end(self, l):
        for c in self._callbacks:
            c.on_episode_end(l)
    def on_step_start(self, l):
        for c in self._callbacks:
            c.on_step_start(l)
    def on_gradients_end(self, l):
        for c in self._callbacks:
            c.on_gradients_end(l)
    def on_epoch_end(self, l):
        for c in self._callbacks:
            c.on_epoch_end(l)


DEFAULT_SAC_CALLBACKS = ComposeCallbacks([
    WandbLoggingCallbacks(),
    VerboseLoggingCallbacks(),
])


##################################################
# SAC

def get_bound_action_fn(action_space, device) -> Callable:
    if isinstance(action_space, gymnasium.spaces.Box):
        min_val = action_space.low
        max_val = action_space.high
        if not np.isfinite(min_val).all() or not np.isfinite(max_val).all():
            # TODO: If any values have infinite domain, then they should not be scaled. Everything else should be scaled.
            raise NotImplementedError()
        scale = (max_val - min_val) / 2
        bias = (max_val + min_val) / 2

        @overload
        def bound_fn(action: np.ndarray) -> np.ndarray: ...
        @overload
        def bound_fn(action: torch.Tensor) -> torch.Tensor: ...

        def bound_fn(action):
            if isinstance(action, np.ndarray):
                return np.tanh(action) * scale + bias # Type: ignore
            if isinstance(action, torch.Tensor):
                return torch.tanh(action) * torch.tensor(scale, device=device) + torch.tensor(bias, device=device) # Type: ignore
            raise ValueError(f'Unrecognized type: {type(action)}')
        return bound_fn
    else:
        raise NotImplementedError()


ModelType = TypeVar('ModelType', FeedforwardModel, RecurrentModel)
class SACTrainer(Trainer, Generic[ModelType]):
    def __init__(
            self,
            env: gymnasium.vector.VectorEnv | gymnasium.Env,
            actor_model: ModelType,
            critic_model_1: ModelType,
            critic_model_2: ModelType,
            actor_optimizer: torch.optim.Optimizer,
            critic_optimizer: torch.optim.Optimizer,
            device: torch.device | None = None,
            config: SACConfig = DEFAULT_SAC_CONFIG):

        if device is None:
            device1 = next(actor_model.parameters()).device
            device2 = next(critic_model_1.parameters()).device
            device3 = next(critic_model_2.parameters()).device
            if device1 != device2 or device1 != device3:
                raise ValueError('Actor and critic model must be on the same device.')
            device = device1

        super().__init__(env, None, None, None, device)

        self.actor_model = actor_model
        self.critic_model_1 = critic_model_1
        self.critic_model_2 = critic_model_2
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.critic_model_target_1 = copy.deepcopy(critic_model_1)
        self.critic_model_target_2 = copy.deepcopy(critic_model_2)

        self._config = config

        self._action_dist_fn_2 = get_action_dist_function(env.action_space)
        self._action_dist_fn = get_action_dist_function(
                env.action_space,
                config={'box': 'squashed'}
        )
        self._bound_action_fn = get_bound_action_fn(env.action_space, device)

    def config(self, key):
        return self._config.get(key, DEFAULT_SAC_CONFIG[key])


##################################################
# Feedforward SAC


class FeedforwardSACTrainer(SACTrainer[FeedforwardModel]):
    def train(self, max_transitions: int | None = None, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS):
        if isinstance(self.env, gymnasium.Env):
            self.train_non_vec(max_transitions=max_transitions, callbacks=callbacks)
        elif isinstance(self.env, gymnasium.vector.VectorEnv):
            self.train_vec(max_transitions=max_transitions, callbacks=callbacks)
        else:
            raise ValueError(f'env must be a gymnasium.Env or gymnasium.vector.VectorEnv. Found {type(self.env)}')

    def train_vec(self, max_transitions: int | None = None, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS):
        callbacks.on_start(locals())

        """
        Loop:
            Collect transitions
            Sample batch
            Gradient step
        """

        num_envs = self.env.num_envs
        history = VecHistoryBuffer(
            max_len = self.config('buffer_size'),
            num_envs = num_envs,
            device = self.device,
        )

        obs, _ = self.env.reset()
        history.append_obs(obs)
        for step in itertools.count():
            if step * num_envs >= max_transitions:
                break

            callbacks.on_step_start(locals())

            if step < self.config('warmup_steps'):
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    model_output = self.actor_model(to_tensor(obs, self.device))

                    action_dist, _ = self._action_dist_fn(model_output)
                    action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore

            history.append_action(action)
            history.append_obs(
                    obs, reward, terminated, truncated
            )

            callbacks.on_transition(locals())

            if terminated.any() or truncated.any():
                callbacks.on_episode_end(locals())

            if step < self.config('warmup_steps'):
                continue

            self._gradient_step(step, history, callbacks)

        callbacks.on_end(locals())

    def train_non_vec(self, max_transitions: int | None = None, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS):
        raise NotImplementedError()

        callbacks.on_start(locals())

        """
        Loop:
            Collect transitions
            Sample batch
            Gradient step
        """

        history = HistoryBuffer(
            max_len = self.config('buffer_size'),
        )

        done = True
        obs = None
        for steps in itertools.count():
            if done:
                obs, _ = self.env.reset()
                done = False

                history.append_obs(obs)
            else:
                with torch.no_grad():
                    model_output = self.actor_model(to_tensor(obs, self.device))

                    action_dist, _ = self._action_dist_fn(model_output)
                    action = action_dist.sample().cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
                done = terminated or truncated

                history.append_action(action)
                history.append_obs(
                        obs, reward, done,
                )

                callbacks.on_transition(locals())


            if done:
                callbacks.on_episode_end(locals())

            if steps < self.config('warmup_steps'):
                continue

            self._gradient_step(history, callbacks)

        callbacks.on_end(locals())

    def _gradient_step(self, step: int, history: HistoryBuffer | VecHistoryBuffer, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS):
        # Sample a random batch
        batch = history.transitions.sample_batch(self.config('batch_size'))

        # Update critic
        y_pred_1 = self.critic_model_1(batch.obs, batch.action)['value']
        y_pred_2 = self.critic_model_2(batch.obs, batch.action)['value']
        with torch.no_grad():
            next_action_dist, next_action_log_prob_fn = self._action_dist_fn(self.actor_model(batch.next_obs))
            next_action = next_action_dist.sample()

            next_state_q = torch.min(
                    self.critic_model_target_1(batch.next_obs, next_action)['value'],
                    self.critic_model_target_2(batch.next_obs, next_action)['value'],
            )
            next_action_log_prob = next_action_dist.log_prob(next_action)
            next_action_log_prob = next_action_log_prob.sum(dim=1, keepdims=True) # type: ignore
            next_action_entropy = self.config('entropy_coeff') * next_action_log_prob
            y_target = batch.reward + self.config('discount') * batch.terminated.logical_not() * (next_state_q - next_action_entropy)

        critic_loss_1 = torch.nn.functional.mse_loss(y_pred_1, y_target)
        critic_loss_2 = torch.nn.functional.mse_loss(y_pred_2, y_target)
        critic_loss = critic_loss_1 + critic_loss_2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        if step % self.config('policy_update_frequency') == 0:
            for _ in range(self.config('policy_update_frequency')):
                action_dist, _ = self._action_dist_fn(self.actor_model(batch.obs))
                action = action_dist.rsample()

                q1 = self.critic_model_1(batch.obs, action)['value']
                q2 = self.critic_model_2(batch.obs, action)['value']
                q = torch.min(q1, q2)

                action_log_prob = action_dist.log_prob(action)
                action_log_prob = action_log_prob.sum(dim=1, keepdims=True) # type: ignore

                actor_loss = q - self.config('entropy_coeff') * action_log_prob
                actor_loss = -actor_loss.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # Update target network
        if step % self.config('target_update_frequency') == 0:
            r = self.config('target_update_rate')
            for p1,p2 in zip(self.critic_model_target_1.parameters(), self.critic_model_1.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data
            for p1,p2 in zip(self.critic_model_target_2.parameters(), self.critic_model_2.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data

        # Callbacks
        callbacks.on_gradients_end(locals())


##################################################
# Recurrent SAC

# TODO
