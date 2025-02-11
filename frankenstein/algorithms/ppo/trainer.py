from collections import OrderedDict, defaultdict
import datetime
import itertools
import os
import time
from typing import Generic, Optional, Callable, TypeVar, TypedDict

import gymnasium
import numpy as np
import tabulate
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn
import torch.nn.utils

from frankenstein.algorithms.utils import recursive_zip
from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss
from frankenstein.algorithms.trainer import Trainer, Checkpoint, NullCheckpoint
from frankenstein.algorithms.utils import to_tensor, reset_hidden, get_action_dist_function, FeedforwardModel, RecurrentModel, format_rate


##################################################
# Config


class PPOConfig(TypedDict, total=False):
    rollout_length: int
    warmup_steps: int
    discount: float
    gae_lambda: float
    norm_adv: bool
    clip_pg_ratio: float
    clip_vf_loss: Optional[float]
    vf_loss_coeff: float
    entropy_loss_coeff: float
    target_kl: Optional[float]
    num_epochs: int
    backtrack: bool
    max_grad_norm: Optional[float]


DEFAULT_PPO_CONFIG = PPOConfig(
    rollout_length = 128,
    warmup_steps = 0,
    discount = 0.99,
    gae_lambda = 0.95,
    norm_adv = True,
    clip_pg_ratio = 0.1,
    clip_vf_loss = None,
    vf_loss_coeff = 0.5,
    entropy_loss_coeff = 0.01,
    target_kl = 0.01,
    num_epochs = 4,
    backtrack = True,
    max_grad_norm = None,
)


##################################################
# Callbacks


class PPOCallbacks():
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


class VerboseLoggingCallbacks(PPOCallbacks):
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

    def on_episode_end(self, l):
        done = l['done']
        self._episode_reward.extend(l['episode_reward'][done])
        self._episode_steps.extend(l['episode_steps'][done])
        self._num_completed_episodes += done.sum()
        self._total_completed_episodes += done.sum()

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
            completed_transitions_total = l['transition_count'] # Total number of transitions
            completed_transitions = l['transition_count'] - l['checkpoint'].start_step # Number of transitions since the last checkpoint
            max_transitions = l['max_transitions'] # Total number of transitions
            print(f'Time: {datetime.datetime.now()}')
            print(f'Completed transitions: {completed_transitions_total:,} ({format_rate(completed_transitions, "step", "steps", time.time() - self._start_time)})')
            print(f'Completed episode(s): {self._num_completed_episodes} ({format_rate(self._num_completed_episodes, "episode", "episodes", time_diff)})')
            if max_transitions is not None:
                remaining_transitions = max_transitions - completed_transitions_total
                transitions_per_second = completed_transitions / (time.time() - self._start_time)
                remaining_seconds = remaining_transitions / transitions_per_second
                print(f'Estimated remaining time: {remaining_seconds:,} seconds')

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
            self._episode_steps.clear()
            self._num_completed_episodes = 0

            if __debug__:
                model = l['self'].model
                weights = torch.tensor([p.mean() for p in model.parameters()])
                print('weights', 'mean', weights.mean(), 'std', weights.std())

            print()


class WandbLoggingCallbacks(PPOCallbacks):
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
            print('Failed to initialize W&B.')
            pass

        self._transition_count = 0
        self._state_values = []
        self._entropy = []
        self._kl = []

        self._data = {}

    def on_start(self, l):
        if self._run is None:
            return
        self._transition_count = l['checkpoint'].start_step
        # TODO: Update config

    def on_transition(self, l):
        if self._run is None:
            return

        self._state_values.append(l['model_output']['value'].mean().item())
        self._entropy.append(l['action_dist'].entropy().mean().item())
        self._transition_count += l['num_envs']

    def on_step_start(self, l):
        if self._run is None:
            return

        if len(self._state_values) > 0:
            self._data['state value'] = np.mean(self._state_values)
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

        done = l['done']
        self._data['episode reward'] = np.mean(l['episode_reward'][done])
        self._data['episode length'] = np.mean(l['episode_steps'][done])

        info = l['info']
        if 'episode' in info and '_episode' in info:
            if info['_episode'].any():
                self._data['true reward'] = np.mean(info['episode']['r'][info['_episode']].tolist())

    def on_epoch_end(self, l):
        if self._run is None:
            return
        self._kl.append(l['losses']['approx_kl'].item())

    def on_gradients_end(self, l):
        if self._run is None:
            return
        self._data['total epochs'] = l['epoch'] + 1
        self._data['kl divergence'] = self._kl[-1] # If we backtrack, then the KL div never gets saved here, so it works regardless of whether we backtrack or not.
        self._kl = []


class ComposeCallbacks(PPOCallbacks):
    def __init__(self, callbacks: list[PPOCallbacks] = []):
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


DEFAULT_PPO_CALLBACKS = ComposeCallbacks([
    WandbLoggingCallbacks(),
    VerboseLoggingCallbacks(),
])


##################################################
# PPO


class IntermediateValues(TypedDict):
    advantages: torch.Tensor
    returns: torch.Tensor
    log_action_probs: torch.Tensor
    state_values: torch.Tensor


def compute_loss_intermediates(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        discount : float,
        gae_lambda : float,
        norm_adv : bool,
        action_dist_fn : Callable,
        compute_model_output_fn : Callable
    ) -> IntermediateValues:
    """
    Calculates some intermediary values for subsequence loss computation:
    - Advantage
    - Returns
    - Log action probabilities
    """
    reward = history.reward
    terminated = history.terminated
    n = len(history.obs_history)

    with torch.no_grad():
        mo= compute_model_output_fn(
            history = history,
            model = model,
            action_dist_fn = action_dist_fn,
        )
        state_values = mo['state_values']
        log_action_probs = mo['log_action_probs']

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values[:n-1,:],
                next_state_values = state_values[1:,:],
                rewards = reward[1:,:],
                terminals = terminated[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )
        returns = advantages + state_values[:n-1,:]

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return IntermediateValues(
        advantages = advantages,
        returns = returns,
        log_action_probs = log_action_probs,
        state_values = state_values,
    )


def compute_ppo_losses(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        intermediate_values : IntermediateValues,
        clip_pg_ratio: float,
        clip_vf_loss : Optional[float],
        action_dist_fn : Callable,
        compute_model_output_fn : Callable):
    """
    Compute the losses for PPO.
    """

    terminated = history.terminated

    n = len(history.obs_history)

    advantages = intermediate_values['advantages']
    returns = intermediate_values['returns']
    log_action_probs_old = intermediate_values['log_action_probs']
    state_values_old = intermediate_values['state_values']

    model_output = compute_model_output_fn(
        history = history,
        model = model,
        action_dist_fn = action_dist_fn,
    )
    state_values = model_output['state_values']
    action_dist = model_output['action_dist']
    log_action_probs = model_output['log_action_probs']
    entropy = action_dist.entropy()

    with torch.no_grad():
        logratio = log_action_probs - log_action_probs_old
        ratio = logratio.exp()
        approx_kl = ((ratio - 1) - logratio).mean()

    # Policy loss
    pg_loss = clipped_advantage_policy_gradient_loss(
            log_action_probs = log_action_probs,
            old_log_action_probs = log_action_probs_old,
            advantages = advantages,
            terminals = terminated[:n-1],
            epsilon=clip_pg_ratio,
    )

    # Value loss
    if clip_vf_loss is not None:
        v_loss_unclipped = (state_values[:n-1] - returns) ** 2
        v_clipped = state_values_old[:n-1] + torch.clamp(
            state_values[:n-1] - state_values_old[:n-1],
            -clip_vf_loss,
            clip_vf_loss,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max
    else:
        v_loss = 0.5 * ((state_values[:n-1] - returns) ** 2)

    entropy_loss = -entropy
    #loss = pg_loss - entropy_loss_coeff * entropy_loss + v_loss * vf_loss_coeff

    return {
            #'loss': loss,
            'loss_pi': pg_loss,
            'loss_vf': v_loss,
            'loss_entropy': entropy_loss,
            'approx_kl': approx_kl,
            'output': model_output['model_output'],
            'hidden': model_output.get('hidden'), # For recurrent models. Returns None otherwise.
    }


ModelType = TypeVar('ModelType', FeedforwardModel, RecurrentModel)
class PPOTrainer(Trainer, Generic[ModelType]):
    def __init__(
            self,
            env: gymnasium.vector.VectorEnv,
            model: ModelType,
            optimizer: torch.optim.Optimizer, # type: ignore
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # type: ignore
            device: torch.device | None = None,
            config: PPOConfig = DEFAULT_PPO_CONFIG):
        if device is None:
            device = next(model.parameters()).device
        super().__init__(env, model, optimizer, scheduler, device)

        self._config = config

        self._action_dist_fn = get_action_dist_function(env.single_action_space)

        if config.get('backtrack') and config.get('target_kl') is None:
            raise ValueError('`target_kl` must be specified when backtracking is enabled.')

    def config(self, key):
        return self._config.get(key, DEFAULT_PPO_CONFIG[key])


##################################################
# Feedforward PPO


def compute_feedforward_model_output(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        action_dist_fn : Callable,
    ):
    # TODO: This is currently only set up to handle tensors and not dicts.

    obs = history.obs
    action = history.action

    device = next(model.parameters()).device
    n = len(history.obs_history)
    num_training_envs = obs[0].shape[0]
    obs_dims = obs[0].shape[1:]

    obs_tensor = to_tensor(obs, device)
    assert isinstance(obs_tensor, torch.Tensor)
    obs_tensor = obs_tensor.view(n * num_training_envs, *obs_dims)
    model_output = model(obs_tensor)
    #model_output = default_collate(model_output)

    reshaped_model_output = {
        **model_output,
        'value': model_output['value'].view(n, num_training_envs),
    }
    if 'action_mean' in model_output:
        reshaped_model_output['action_mean'] = model_output['action_mean'].view(n, num_training_envs, -1)
        reshaped_model_output['action_logstd'] = model_output['action_logstd'].view(n, num_training_envs, -1)
    elif 'action' in model_output:
        reshaped_model_output['action'] = model_output['action'].view(n, num_training_envs, -1)

    assert 'value' in model_output
    state_values = reshaped_model_output['value']
    action_dist, action_dist_log_prob = action_dist_fn(reshaped_model_output,n-1)
    log_action_probs = action_dist_log_prob(action)

    return {
        'model_output': model_output,
        'state_values': state_values,
        'action_dist': action_dist,
        'action_dist_log_prob': action_dist_log_prob,
        'log_action_probs': log_action_probs,
    }


class FeedforwardPPOTrainer(PPOTrainer[FeedforwardModel]):
    def train(self, max_transitions: int | None = None, callbacks: PPOCallbacks = DEFAULT_PPO_CALLBACKS, checkpoint: Checkpoint | None = None):
        if checkpoint is None:
            checkpoint = NullCheckpoint()

        callbacks.on_start(locals())

        num_envs = self.env.num_envs

        history = VecHistoryBuffer(
                num_envs = num_envs,
                max_len=self.config('rollout_length')+1,
                device=self.device)

        episode_reward = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs)

        obs, info = self.env.reset()
        history.append_obs(obs)

        ##################################################
        # Warmup
        # The state of all environments are similar at the start of an episode. By warming up, we increase the diversity of states to something that is closer to iid.

        if self.config('warmup_steps') > 0:
            callbacks.on_warmup_start(locals())

        start_time = time.time()
        warmup_episode_rewards = defaultdict(lambda: [])
        warmup_episode_steps = defaultdict(lambda: [])
        for _ in range(self.config('warmup_steps')):
            # Select action
            with torch.no_grad():
                model_output = self.model(to_tensor(obs, self.device))

                action_dist, _ = self._action_dist_fn(model_output)
                action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated | truncated

            episode_reward += reward
            episode_steps += 1

            if done.any():
                # Reset episode stats
                episode_reward[done] = 0
                episode_steps[done] = 0

        if self.config('warmup_steps') > 0:
            print(f'Warmup time: {time.time() - start_time:.2f} s')
            callbacks.on_warmup_end(locals())

        ##################################################
        # Start training
        callbacks.on_training_start(locals())

        start_step = checkpoint.start_step // (num_envs * self.config('rollout_length'))
        transition_count = checkpoint.start_step
        for step in itertools.count(start_step):
            transition_count = step * num_envs * self.config('rollout_length')
            checkpoint.save(transition_count)

            if max_transitions is not None and transition_count >= max_transitions:
                break

            callbacks.on_step_start(locals())

            # Gather data
            callbacks.on_gather_start(locals())
            episode_rewards = [] # For multitask weighing purposes
            for _ in range(self.config('rollout_length')):
                # Select action
                with torch.no_grad():
                    model_output = self.model(to_tensor(obs, self.device))

                    action_dist, _ = self._action_dist_fn(model_output)
                    action = action_dist.sample().cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
                done = terminated | truncated

                callbacks.on_transition(locals())

                history.append_action(action)
                episode_steps += 1
                episode_reward += reward

                history.append_obs(
                        obs, reward, done,
                )

                if done.any():
                    callbacks.on_episode_end(locals())
                    # Save episode rewards
                    for r in episode_reward[done]:
                        episode_rewards.append(r)
                    # Reset episode stats
                    episode_reward[done] = 0
                    episode_steps[done] = 0
            callbacks.on_gather_end(locals())

            # Train
            interm = self.compute_loss_intermediates(
                    history = history,
            )
            state_dict = None
            num_transitions = num_envs * self.config('rollout_length')
            num_transitions_clipped = num_transitions // self.config('num_epochs') * self.config('num_epochs')
            minibatch_indices = torch.randperm(num_transitions)[:num_transitions_clipped].reshape(self.config('num_epochs'), -1)
            for epoch in range(self.config('num_epochs')):
                losses = self.compute_ppo_losses(
                        history = history,
                        intermediate_values = interm,
                )

                # If we surpass the KL target, return to the last known weights that did not surpass the KL target
                t_kl = self.config('target_kl')
                if t_kl is not None:
                    if losses['approx_kl'] > t_kl:
                        if state_dict is not None:
                            self.model.load_state_dict(state_dict[0])
                            self.optimizer.load_state_dict(state_dict[1])
                        break
                    if self.config('backtrack'):
                        state_dict = (
                            self.model.state_dict(),
                            self.optimizer.state_dict(),
                        )

                vf_loss_coeff = self.config('vf_loss_coeff')
                entropy_loss_coeff = self.config('entropy_loss_coeff')

                # Select a batch of data and perform a gradient step
                idx = minibatch_indices[epoch,:]
                loss_pi = losses['loss_pi'].flatten()[idx].mean()
                loss_vf = losses['loss_vf'].flatten()[idx].mean()
                loss_entropy = losses['loss_entropy'].flatten()[idx].mean()

                loss = loss_pi + vf_loss_coeff * loss_vf + entropy_loss_coeff * loss_entropy

                if not torch.isfinite(loss):
                    raise ValueError('Invalid loss computed')

                self.optimizer.zero_grad()
                loss.backward()
                if self.config('max_grad_norm') is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config('max_grad_norm'))
                self.optimizer.step()

                callbacks.on_epoch_end(locals())

            callbacks.on_gradients_end(locals())

            # Clear data
            history.clear()
            episode_rewards = []

        if transition_count != checkpoint.start_step:
            checkpoint.save(transition_count, force=True)

        callbacks.on_end(locals())

    def compute_loss_intermediates(
            self,
            history : VecHistoryBuffer) -> IntermediateValues:
        return compute_loss_intermediates(
                history = history,
                model = self.model,
                discount = self.config('discount'),
                gae_lambda = self.config('gae_lambda'),
                norm_adv = self.config('norm_adv'),
                action_dist_fn = self._action_dist_fn,
                compute_model_output_fn = compute_feedforward_model_output,
        )

    def compute_ppo_losses(
            self,
            history : VecHistoryBuffer,
            intermediate_values : IntermediateValues):
        return compute_ppo_losses(
                history = history,
                model = self.model,
                intermediate_values = intermediate_values,
                clip_pg_ratio = self.config('clip_pg_ratio'),
                clip_vf_loss = self.config('clip_vf_loss'),
                action_dist_fn = self._action_dist_fn,
                compute_model_output_fn = compute_feedforward_model_output,
        )


##################################################
# Recurrent PPO


def compute_recurrent_model_output(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        action_dist_fn : Callable,
    ):

    obs = history.obs
    action = history.action
    terminated = history.terminated
    misc = history.misc
    assert isinstance(misc,dict)
    hidden = misc['hidden']

    device = next(model.parameters()).device
    num_training_envs = len(terminated[0])
    n = len(history.obs_history)

    model_output = []
    curr_hidden = tuple([h[0].detach() for h in hidden])
    initial_hidden = model.init_hidden(num_training_envs) # type: ignore
    for o,term in recursive_zip(obs,terminated):
        curr_hidden = reset_hidden(
                terminal = term,
                hidden = curr_hidden,
                initial_hidden = initial_hidden,
                batch_dim = model.hidden_batch_dims,
        )
        o = to_tensor(o, device)
        mo = model(o,curr_hidden)
        curr_hidden = mo['hidden']
        model_output.append(mo)
    model_output = default_collate(model_output)

    assert 'value' in model_output
    state_values = model_output['value'].squeeze()
    action_dist, action_dist_log_prob = action_dist_fn(model_output,n-1)
    log_action_probs = action_dist_log_prob(action)

    return {
        'model_output': model_output,
        'state_values': state_values,
        'action_dist': action_dist,
        'action_dist_log_prob': action_dist_log_prob,
        'log_action_probs': log_action_probs,
        'hidden': tuple(h.detach() for h in curr_hidden),
    }


class RecurrentPPOTrainer(PPOTrainer[RecurrentModel]):
    def train(self, max_transitions: int | None = None, callbacks: PPOCallbacks = DEFAULT_PPO_CALLBACKS, checkpoint: Checkpoint | None = None):
        if checkpoint is None:
            checkpoint = NullCheckpoint()

        callbacks.on_start(locals())

        num_envs = self.env.num_envs

        history = VecHistoryBuffer(
                num_envs = num_envs,
                max_len=self.config('rollout_length')+1,
                device=self.device)

        episode_reward = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs)

        obs, info = self.env.reset()
        hidden = self.model.init_hidden(num_envs) # type: ignore (???)
        history.append_obs(
                #{k:v for k,v in obs.items() if k not in obs_ignore},
                obs,
                misc = {'hidden': hidden, 't': episode_steps.copy()},
        )

        ##################################################
        # Warmup
        # The state of all environments are similar at the start of an episode. By warming up, we increase the diversity of states to something that is closer to iid.

        if self.config('warmup_steps') > 0:
            callbacks.on_warmup_start(locals())

        start_time = time.time()
        warmup_episode_rewards = defaultdict(lambda: [])
        warmup_episode_steps = defaultdict(lambda: [])
        for _ in range(self.config('warmup_steps')):
            # Select action
            with torch.no_grad():
                model_output = self.model(to_tensor(obs, self.device), hidden)
                hidden = model_output['hidden']

                action_dist, _ = self._action_dist_fn(model_output)
                action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
            done = terminated | truncated

            episode_reward += reward
            episode_steps += 1

            if done.any():
                # Reset hidden state for finished episodes
                hidden = reset_hidden(
                        terminal = torch.tensor(done, device=self.device),
                        hidden = hidden,
                        initial_hidden = self.model.init_hidden(num_envs),
                        batch_dim = self.model.hidden_batch_dims,
                )
                # Reset episode stats
                episode_reward[done] = 0
                episode_steps[done] = 0

        if self.config('warmup_steps') > 0:
            print(f'Warmup time: {time.time() - start_time:.2f} s')
            callbacks.on_warmup_end(locals())

        ##################################################
        # Start training
        callbacks.on_training_start(locals())

        start_step = checkpoint.start_step // (num_envs * self.config('rollout_length'))
        transition_count = checkpoint.start_step
        for step in itertools.count(start_step):
            transition_count = step * num_envs * self.config('rollout_length')
            checkpoint.save(transition_count)

            if max_transitions is not None and transition_count >= max_transitions:
                break

            callbacks.on_step_start(locals())

            # Gather data
            callbacks.on_gather_start(locals())
            episode_rewards = [] # For multitask weighing purposes
            for _ in range(self.config('rollout_length')):
                # Select action
                with torch.no_grad():
                    model_output = self.model(to_tensor(obs, self.device), hidden)
                    hidden = model_output['hidden']

                    action_dist, _ = self._action_dist_fn(model_output)
                    action = action_dist.sample().cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action) # type: ignore
                done = terminated | truncated

                callbacks.on_transition(locals())

                history.append_action(action)
                episode_steps += 1
                episode_reward += reward

                history.append_obs(
                        obs, reward, done,
                        misc = {'hidden': hidden, 't': episode_steps.copy()}
                )

                if done.any():
                    callbacks.on_episode_end(locals())

                    # Reset hidden state for finished episodes
                    hidden = reset_hidden(
                        terminal = torch.tensor(done, device=self.device),
                        hidden = hidden,
                        initial_hidden = self.model.init_hidden(num_envs),
                        batch_dim = self.model.hidden_batch_dims,
                    )
                    # Save episode rewards
                    for r in episode_reward[done]:
                        episode_rewards.append(r)
                    # Reset episode stats
                    episode_reward[done] = 0
                    episode_steps[done] = 0
            callbacks.on_gather_end(locals())

            # Train
            interm = self.compute_loss_intermediates(
                    history = history,
            )
            state_dict = None
            num_transitions = num_envs * self.config('rollout_length')
            num_transitions_clipped = num_transitions // self.config('num_epochs') * self.config('num_epochs')
            minibatch_indices = torch.randperm(num_transitions)[:num_transitions_clipped].reshape(self.config('num_epochs'), -1)
            for epoch in range(self.config('num_epochs')):
                losses = self.compute_ppo_losses(
                        history = history,
                        intermediate_values = interm,
                )

                # If we surpass the KL target, return to the last known weights that did not surpass the KL target
                t_kl = self.config('target_kl')
                if t_kl is not None:
                    if losses['approx_kl'] > t_kl:
                        if state_dict is not None:
                            self.model.load_state_dict(state_dict[0])
                            self.optimizer.load_state_dict(state_dict[1])
                        break
                    if self.config('backtrack'):
                        state_dict = (
                            self.model.state_dict(),
                            self.optimizer.state_dict(),
                        )

                vf_loss_coeff = self.config('vf_loss_coeff')
                entropy_loss_coeff = self.config('entropy_loss_coeff')

                # Select a batch of data and perform a gradient step
                idx = minibatch_indices[epoch,:]
                loss_pi = losses['loss_pi'].flatten()[idx].mean()
                loss_vf = losses['loss_vf'].flatten()[idx].mean()
                loss_entropy = losses['loss_entropy'].flatten()[idx].mean()
                #loss_pi = losses['loss_pi'].mean()
                #loss_vf = losses['loss_vf'].mean()
                #loss_entropy = losses['loss_entropy'].mean()

                loss = loss_pi + vf_loss_coeff * loss_vf + entropy_loss_coeff * loss_entropy

                if not torch.isfinite(loss):
                    raise ValueError('Invalid loss computed')

                self.optimizer.zero_grad()
                loss.backward()
                if self.config('max_grad_norm') is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config('max_grad_norm'))
                self.optimizer.step()

                callbacks.on_epoch_end(locals())

            callbacks.on_gradients_end(locals())

            # Clear data
            history.clear()
            episode_rewards = []

        if transition_count != checkpoint.start_step:
            checkpoint.save(transition_count, force=True)

        callbacks.on_end(locals())

    def compute_loss_intermediates(
            self,
            history : VecHistoryBuffer,
        ) -> IntermediateValues:
            return compute_loss_intermediates(
                    history = history,
                    model = self.model,
                    discount = self.config('discount'),
                    gae_lambda = self.config('gae_lambda'),
                    norm_adv = self.config('norm_adv'),
                    action_dist_fn = self._action_dist_fn,
                    compute_model_output_fn = compute_recurrent_model_output,
            )

    def compute_ppo_losses(
            self,
            history : VecHistoryBuffer,
            intermediate_values : IntermediateValues,
            ):
        return compute_ppo_losses(
                history = history,
                model = self.model,
                intermediate_values = intermediate_values,
                clip_pg_ratio = self.config('clip_pg_ratio'),
                clip_vf_loss = self.config('clip_vf_loss'),
                action_dist_fn = self._action_dist_fn,
                compute_model_output_fn = compute_recurrent_model_output,
        )
