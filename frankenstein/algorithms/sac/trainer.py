from collections import OrderedDict, defaultdict, namedtuple
import copy
import datetime
import itertools
import os
import time
from typing import Generic, TypeVar, TypedDict, Callable, overload

import gymnasium
import numpy as np
import numpy.typing as npt
import tabulate
import torch
import torch.nn
import torch.nn.utils

from frankenstein.buffer import HistoryBuffer
from frankenstein.algorithms.trainer import Trainer
from frankenstein.algorithms.utils import to_tensor, get_action_dist_function, FeedforwardModel, RecurrentModel, format_rate, recursive_zip, reset_hidden
from frankenstein.buffer.vec_history import VecHistoryBuffer, NumpyBackedVecHistoryBuffer, SerializeFn, make_default_serde
from frankenstein.algorithms.trainer import Trainer, Checkpoint, NullCheckpoint
from frankenstein.buffer.views import TrajectoryBatch


##################################################
# Config


class SACConfig(TypedDict, total=False):
    buffer_size: int
    warmup_steps: int # Number of transitions to collect before training starts
    batch_size: int
    discount: float
    policy_update_interval: int # Number of gradient steps between policy network updates
    target_update_interval: int # Number of gradient steps between target network updates
    target_update_rate: float # Polyak averaging rate for target network updates. 1 means hard update. 0 means no update.
    entropy_coeff: float
    trajectory_length: int


DEFAULT_SAC_CONFIG = SACConfig(
    buffer_size = 1_000_000,
    warmup_steps = 5_000,
    batch_size = 256,
    discount = 0.99,
    policy_update_interval=2,
    target_update_interval=1,
    target_update_rate=0.005,
    entropy_coeff = 0.05,
    trajectory_length = 128,
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
        if 'episode' in info and '_episode' in info:
            if info['_episode'].any():
                self._episode_true_reward.extend(info['episode']['r'][info['_episode']].tolist())

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
        if 'episode' in info and '_episode' in info:
            if info['_episode'].any():
                self._data['true reward'] = np.mean(info['episode']['r'][info['_episode']].tolist())


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


def compute_recurrent_model_output(actor_model: RecurrentModel, critic_models: list[RecurrentModel], batch: TrajectoryBatch):
    """
    Args:
        critic_models: A list of critic models in the order `critic_model_1`, `critic_model_2`, `critic_model_target_1`, `critic_model_target_2`.
    """
    obs = batch.obs
    action = batch.action
    terminated = batch.terminated

    actor_hidden = batch.misc[0]['actor_hidden']
    critic_hiddens = batch.misc[0]['critic_hidden']

    device = next(actor_model.parameters()).device
    num_training_envs = len(terminated[0])
    n = len(batch.obs)

    actor_model_output = []
    actor_curr_hidden = tuple([h[0].detach() for h in actor_hidden])
    actor_initial_hidden = actor_model.init_hidden(num_training_envs)
    critic_model_outputs = [[] for _ in critic_models]
    critic_curr_hiddens = [tuple([h[0].detach() for h in critic_hidden]) for critic_hidden in critic_hiddens]
    critic_initial_hiddens = [critic_model.init_hidden(num_training_envs) for critic_model in critic_models]
    for o,term in recursive_zip(obs,terminated):
        o = to_tensor(o, device)
        # Actor
        actor_curr_hidden = reset_hidden(
                terminal = term,
                hidden = actor_curr_hidden,
                initial_hidden = actor_initial_hidden,
                batch_dim = actor_model.hidden_batch_dims,
        )
        mo = actor_model(o,actor_curr_hidden)
        actor_curr_hidden = mo['hidden']
        actor_model_output.append(mo)
        # Critics
        new_critic_curr_hiddens = []
        for critic_model, critic_curr_hidden, critic_initial_hidden, critic_model_output in zip(critic_models, critic_curr_hiddens, critic_initial_hiddens, critic_model_outputs):
            new_critic_curr_hiddens.append(reset_hidden(
                terminal = term,
                hidden = critic_curr_hidden,
                initial_hidden = critic_initial_hidden,
                batch_dim = critic_model.hidden_batch_dims,
            ))
            mo = critic_model(o,critic_curr_hidden)
            new_critic_curr_hiddens.append(mo['hidden'])
            critic_model_output.append(mo)
        critic_curr_hiddens = new_critic_curr_hiddens
    #model_output = default_collate(model_output)

    return {
        'values': [...],
        'hidden': ...,
    }


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
                env.single_action_space if isinstance(env, gymnasium.vector.VectorEnv) else env.action_space,
                config={'box': 'squashed'}
        )
        self._bound_action_fn = get_bound_action_fn(env.action_space, device)

    def config(self, key):
        return self._config.get(key, DEFAULT_SAC_CONFIG[key])


##################################################
# Feedforward SAC


class FeedforwardSACTrainer(SACTrainer[FeedforwardModel]):
    def train(self, max_transitions: int | None = None, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS, checkpoint: Checkpoint | None = None):
        if checkpoint is None:
            checkpoint = NullCheckpoint()

        if isinstance(self.env, gymnasium.Env):
            self.train_non_vec(max_transitions=max_transitions, callbacks=callbacks, checkpoint=checkpoint)
        elif isinstance(self.env, gymnasium.vector.VectorEnv):
            self.train_vec(max_transitions=max_transitions, callbacks=callbacks, checkpoint=checkpoint)
        else:
            raise ValueError(f'env must be a gymnasium.Env or gymnasium.vector.VectorEnv. Found {type(self.env)}')

    def train_vec(self, max_transitions: int | None, callbacks: SACCallbacks, checkpoint: Checkpoint):
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
        start_step = checkpoint.start_step // num_envs
        transition_count = checkpoint.start_step
        for step in itertools.count(start_step):
            transition_count = step * num_envs
            checkpoint.save(transition_count)

            if transition_count >= max_transitions:
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

        if transition_count != checkpoint.start_step:
            checkpoint.save(transition_count, force=True)

        callbacks.on_end(locals())

    def train_non_vec(self, max_transitions: int | None, callbacks: SACCallbacks, checkpoint: Checkpoint):
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
        if step % self.config('policy_update_interval') == 0:
            for _ in range(self.config('policy_update_interval')):
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
        if step % self.config('target_update_interval') == 0:
            r = self.config('target_update_rate')
            for p1,p2 in zip(self.critic_model_target_1.parameters(), self.critic_model_1.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data
            for p1,p2 in zip(self.critic_model_target_2.parameters(), self.critic_model_2.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data

        # Callbacks
        callbacks.on_gradients_end(locals())


##################################################
# Recurrent SAC


CriticModels = namedtuple('CriticModels', [
    'critic_model_1',
    'critic_model_2',
    'critic_model_target_1',
    'critic_model_target_2',
])


#class MiscWrapper:
#    def __init__(self, value):
#        self._value = value
#
#    def __getitem__(self, index):
#        return {
#            'actor_hidden':  tuple(x.index_select(0, torch.tensor([index], device=x.device)).squeeze(0) for x in self._value['actor_hidden']),
#            'critic_hidden': CriticModels._make(
#                tuple(x.index_select(0, torch.tensor([index], device=x.device)).squeeze(0) for x in hidden)
#                for hidden in self._value['critic_hidden']
#            ),
#        }


def make_serde() -> dict[str, SerializeFn]:
    """
    Create default serialization/deserialization functions but replace the misc serialization/deserialization with one that can handle the hidden states of the recurrent models.
    """
    default_serde = make_default_serde()
    dtype = None
    indices = None
    def serialize_fn(misc: dict) -> npt.NDArray[np.uint8]:
        """
        Args:
            misc: A dictionary containing the hidden states of the actor and critic models.
                - 'actor_hidden': A tuple of tensors containing the hidden state of the actor model. Each tensor has shape (num_envs, *hidden_dim).
                - 'critic_hidden': A namedtuple where each element corresponds to one of the four critic models (critic_model_1, critic_model_2, critic_model_target_1, critic_model_target_2). Each element is a tuple of tensors containing the hidden state of the corresponding critic model. Each tensor has shape (num_envs, *hidden_dim).
        """
        nonlocal dtype, indices
        tensors = [
            *misc['actor_hidden'],
            *misc['critic_hidden'].critic_model_1,
            *misc['critic_hidden'].critic_model_2,
            *misc['critic_hidden'].critic_model_target_1,
            *misc['critic_hidden'].critic_model_target_2
        ]
        serialized = torch.cat(tensors, dim=1)
        if dtype is None:
            dtype = serialized.dtype
            # Dim 0: num_envs
            # Dim 1,...: hidden state shape
            indices = {
                'actor_hidden': tuple(torch.tensor(x.shape[1:]).sum().item() for x in misc['actor_hidden']),
                'critic_hidden': CriticModels._make(
                    tuple(torch.tensor(x.shape[1:]).sum().item() for x in hidden)
                    for hidden in misc['critic_hidden']
                ),
            }
        return serialized.cpu().numpy().view(np.uint8)
    def deserialize_transition_fn(data: npt.NDArray[np.uint8]) -> dict:
        raise NotImplementedError()
    def deserialize_trajectory_fn(data: npt.NDArray[np.uint8]) -> dict:
        nonlocal dtype, indices
        assert len(data.shape) == 3, 'Expected data to have shape (batch_size, trajectory_length, serialized_misc_dim)'
        assert dtype is not None and indices is not None, 'deserialize_fn called before serialize_fn'
        sizes = [
            *indices['actor_hidden'],
             *indices['critic_hidden'].critic_model_1,
             *indices['critic_hidden'].critic_model_2,
             *indices['critic_hidden'].critic_model_target_1,
             *indices['critic_hidden'].critic_model_target_2,
        ]
        tensors = torch.from_numpy(data).view(dtype).split(sizes, dim=2)
        tensor_iter = iter(tensors)
        return {
            'actor_hidden': tuple(next(tensor_iter) for _ in indices['actor_hidden']),
            'critic_hidden': CriticModels(
                critic_model_1 = tuple(next(tensor_iter) for _ in indices['critic_hidden'].critic_model_1),
                critic_model_2 = tuple(next(tensor_iter) for _ in indices['critic_hidden'].critic_model_2),
                critic_model_target_1 = tuple(next(tensor_iter) for _ in indices['critic_hidden'].critic_model_target_1),
                critic_model_target_2 = tuple(next(tensor_iter) for _ in indices['critic_hidden'].critic_model_target_2),
            ),
        }
    return {
        'serialize_fn': SerializeFn(
            obs = default_serde[0].obs,
            action = default_serde[0].action,
            reward = default_serde[0].reward,
            misc = serialize_fn,
        ),
        'deserialize_transition_fn': SerializeFn(
            obs = default_serde[1].obs,
            action = default_serde[1].action,
            reward = default_serde[1].reward,
            misc = deserialize_transition_fn,
        ),
        'deserialize_trajectory_fn': SerializeFn(
            obs = default_serde[2].obs,
            action = default_serde[2].action,
            reward = default_serde[2].reward,
            misc = deserialize_trajectory_fn,
        ),
    }


class RecurrentSACTrainer(SACTrainer[RecurrentModel]):
    def train(self, max_transitions: int | None = None, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS, checkpoint: Checkpoint | None = None):
        if checkpoint is None:
            checkpoint = NullCheckpoint()

        if isinstance(self.env, gymnasium.Env):
            self.train_non_vec(max_transitions=max_transitions, callbacks=callbacks, checkpoint=checkpoint)
        elif isinstance(self.env, gymnasium.vector.VectorEnv):
            self.train_vec(max_transitions=max_transitions, callbacks=callbacks, checkpoint=checkpoint)
        else:
            raise ValueError(f'env must be a gymnasium.Env or gymnasium.vector.VectorEnv. Found {type(self.env)}')

    def train_vec(self, max_transitions: int | None, callbacks: SACCallbacks, checkpoint: Checkpoint):
        callbacks.on_start(locals())

        """
        Loop:
            Collect transitions
            Sample batch
            Gradient step
        """

        if max_transitions is not None and max_transitions < 0:
            max_transitions = None

        num_envs = self.env.num_envs
        #history = VecHistoryBuffer(
        #    max_len = self.config('buffer_size'),
        #    num_envs = num_envs,
        #    trajectory_length = self.config('trajectory_length'),
        #    device = self.device,
        #)
        history = NumpyBackedVecHistoryBuffer(
            max_len = self.config('buffer_size'),
            num_envs = num_envs,
            trajectory_length = self.config('trajectory_length'),
            **make_serde(), # type: ignore
        )
        self.history = history

        actor_hidden = self.actor_model.init_hidden(num_envs)
        critic_models = CriticModels(
                critic_model_1 = self.critic_model_1,
                critic_model_2 = self.critic_model_2,
                critic_model_target_1 = self.critic_model_target_1,
                critic_model_target_2 = self.critic_model_target_2,
        )
        critic_hidden = CriticModels._make([m.init_hidden(num_envs) for m in critic_models])

        obs, _ = self.env.reset()
        #history.append_obs(obs, misc=MiscWrapper({'actor_hidden': actor_hidden, 'critic_hidden': critic_hidden}))
        history.append_obs(obs, misc={'actor_hidden': actor_hidden, 'critic_hidden': critic_hidden})
        start_step = checkpoint.start_step // num_envs
        transition_count = checkpoint.start_step
        for step in itertools.count(start_step):
            transition_count = step * num_envs
            checkpoint.save(transition_count)

            if max_transitions is not None and transition_count >= max_transitions:
                break

            callbacks.on_step_start(locals())

            if step < self.config('warmup_steps'):
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = to_tensor(obs, self.device)
                    model_output = self.actor_model(obs_tensor, actor_hidden)

                    action_dist, _ = self._action_dist_fn(model_output)
                    action = action_dist.sample().cpu().numpy()
                    action_tensor = to_tensor(action, self.device)

                    actor_hidden = model_output['hidden']
                    critic_hidden = CriticModels._make([
                            m(obs_tensor, a, h)['hidden']
                            for m,a,h in zip(critic_models, [action_tensor]*len(CriticModels._fields), critic_hidden)
                    ])

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore

            history.append_action(action)
            history.append_obs(
                    obs, reward, terminated, truncated,
                    #misc=MiscWrapper({'actor_hidden': actor_hidden, 'critic_hidden': critic_hidden}),
                    misc={'actor_hidden': actor_hidden, 'critic_hidden': critic_hidden},
            )

            callbacks.on_transition(locals())

            if terminated.any() or truncated.any():
                callbacks.on_episode_end(locals())

            if step < self.config('warmup_steps'):
                continue

            self._gradient_step(step, history, callbacks)

        if transition_count != checkpoint.start_step:
            checkpoint.save(transition_count, force=True)

        callbacks.on_end(locals())

    def train_non_vec(self, max_transitions: int | None, callbacks: SACCallbacks, checkpoint: Checkpoint):
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

    def _gradient_step(self, step: int, history: HistoryBuffer | VecHistoryBuffer | NumpyBackedVecHistoryBuffer, callbacks: SACCallbacks = DEFAULT_SAC_CALLBACKS):
        # Sample a random batch
        batch = history.trajectories.sample_batch(self.config('batch_size'))
        batch_obs = batch.obs.permute(1,0,2).to(self.device) # [trajectory_length, batch, ...]
        batch_action = batch.action.permute(1,0,2).to(self.device) # [trajectory_length, batch, ...]
        batch_next_obs = batch.next_obs.permute(1,0,2).to(self.device) # [trajectory_length, batch, ...]
        batch_reward = batch.reward.permute(1,0).float().to(self.device) # [trajectory_length, batch]
        batch_terminated = batch.terminated.permute(1,0).to(self.device) # [trajectory_length, batch]

        # Update critic
        def get_first_critic_hidden(batch_misc, key, model_name):
            #return default_collate([getattr(m[0][key],model_name) for m in batch_misc])
            return tuple(
                m[:,0,:].to(self.device) for m in getattr(batch_misc[key], model_name)
            )
        def get_first_actor_hidden(batch_misc):
            #return default_collate([m[0]['actor_hidden'] for m in batch_misc])
            return tuple(
                m[:,0,:].to(self.device) for m in batch_misc['actor_hidden']
            )
        def compute_critic_output(model: RecurrentModel, obs, action, hidden):
            """
            Args:
                obs: [trajectory_length, batch, *obs_shape]
                action: [trajectory_length, batch, *action_shape]
                hidden: [batch, ...]
            Returns: value: [trajectory_length, batch, 1]
            """
            #outputs = []
            #for o,a in zip(obs.unbind(0), action.unbind(0)): # dims: [trajectory_length, batch, ...]
            #    o = to_tensor(o, self.device) # [batch, *obs_shape]
            #    a = to_tensor(a, self.device) # [batch, *action_shape]
            #    mo = model(o,a,hidden)
            #    hidden = mo['hidden']
            #    outputs.append(mo['value'].squeeze(1)) # [batch, 1] -> [batch]
            #return torch.stack([o for o in outputs], dim=0)
            output = model.forward_sequence(obs, action, hidden=hidden)
            return output['value']
        def sample_actions(model: RecurrentModel, obs, hidden):
            """
            Output shape: [trajectory_length, batch, action_dim]
            """
            #actions = []
            #action_log_probs = []
            #for o in obs.unbind(0): # dims: [trajectory_length, batch, ...]
            #    o = to_tensor(o, self.device)
            #    mo = model(o,hidden)
            #    hidden = mo['hidden']
            #    action_dist, _ = self._action_dist_fn(mo)
            #    action = action_dist.rsample()
            #    actions.append(action)
            #    action_log_probs.append(action_dist.log_prob(action).sum(dim=1, keepdims=False)) # type: ignore
            #return torch.stack(actions, dim=0), torch.stack(action_log_probs, dim=0)
            output = model.forward_sequence(obs, hidden=hidden)
            action_dist, _ = self._action_dist_fn(output)
            action = action_dist.rsample()
            action_log_prob = action_dist.log_prob(action)
            action_log_prob = action_log_prob.sum(dim=2, keepdims=False) # type: ignore
            return action, action_log_prob
        y_pred_1 = compute_critic_output(
                self.critic_model_1, batch_obs, batch_action,
                get_first_critic_hidden(batch.misc, 'critic_hidden', 'critic_model_1')
        )
        y_pred_2 = compute_critic_output(
                self.critic_model_2, batch_obs, batch_action,
                get_first_critic_hidden(batch.misc, 'critic_hidden', 'critic_model_2')
        )
        with torch.no_grad():
            next_action, next_action_log_prob = sample_actions(
                    self.actor_model, batch_next_obs,
                    get_first_actor_hidden(batch.next_misc),
            )

            next_state_q = torch.min(
                    compute_critic_output(
                        self.critic_model_target_1, batch_next_obs, next_action,
                        get_first_critic_hidden(batch.next_misc, 'critic_hidden', 'critic_model_target_1')
                    ),
                    compute_critic_output(
                        self.critic_model_target_2, batch_next_obs, next_action,
                        get_first_critic_hidden(batch.next_misc, 'critic_hidden', 'critic_model_target_2')
                    ),
            )
            next_action_entropy = self.config('entropy_coeff') * next_action_log_prob
            y_target = batch_reward + self.config('discount') * batch_terminated.logical_not() * (next_state_q - next_action_entropy)

        assert y_pred_1.shape == y_target.shape == y_pred_2.shape, f'Shape mismatch: {y_pred_1.shape} {y_target.shape} {y_pred_2.shape}'
        critic_loss_1 = torch.nn.functional.mse_loss(y_pred_1, y_target)
        critic_loss_2 = torch.nn.functional.mse_loss(y_pred_2, y_target)
        critic_loss = critic_loss_1 + critic_loss_2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        if step % self.config('policy_update_interval') == 0:
            for _ in range(self.config('policy_update_interval')):
                action, action_log_prob = sample_actions(
                        self.actor_model, batch_obs,
                        get_first_actor_hidden(batch.misc),
                )

                q1 = compute_critic_output(
                        self.critic_model_1, batch_obs, action,
                        get_first_critic_hidden(batch.misc, 'critic_hidden', 'critic_model_1'),
                )
                q2 = compute_critic_output(
                        self.critic_model_2, batch_obs, action,
                        get_first_critic_hidden(batch.misc, 'critic_hidden', 'critic_model_2'),
                )
                q = torch.min(q1, q2)

                actor_loss = q - self.config('entropy_coeff') * action_log_prob
                actor_loss = -actor_loss.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # Update target network
        if step % self.config('target_update_interval') == 0:
            r = self.config('target_update_rate')
            for p1,p2 in zip(self.critic_model_target_1.parameters(), self.critic_model_1.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data
            for p1,p2 in zip(self.critic_model_target_2.parameters(), self.critic_model_2.parameters()):
                p1.data = r*p2.data + (1-r)*p1.data

        # Callbacks
        callbacks.on_gradients_end(locals())

    def state_dict(self):
        return {
            'actor_model': self.actor_model.state_dict(),
            'critic_model_1': self.critic_model_1.state_dict(),
            'critic_model_2': self.critic_model_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic_model_target_1': self.critic_model_target_1.state_dict(),
            'critic_model_target_2': self.critic_model_target_2.state_dict(),
            'buffer': self.history.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor_model.load_state_dict(state_dict['actor_model'])
        self.critic_model_1.load_state_dict(state_dict['critic_model_1'])
        self.critic_model_2.load_state_dict(state_dict['critic_model_2'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.critic_model_target_1.load_state_dict(state_dict['critic_model_target_1'])
        self.critic_model_target_2.load_state_dict(state_dict['critic_model_target_2'])
        self.history.load_state_dict(state_dict['buffer'])
