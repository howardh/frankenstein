from typing import Sequence

import gymnasium
import numpy as np
import torch
from torch.types import _size

def recursive_zip(*args):
    """
    Zip objects together. If dictionaries are provided, the lists within the dictionary are zipped together.

    >>> list(recursive_zip([1,2,3], [4,5,6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(recursive_zip({'a': [4,5,6], 'b': [7,8,9]}))
    [{'a': 4, 'b': 7}, {'a': 5, 'b': 8}, {'a': 6, 'b': 9}]

    >>> list(recursive_zip([1,2,3], {'a': [4,5,6], 'b': [7,8,9]}))
    [(1, {'a': 4, 'b': 7}), (2, {'a': 5, 'b': 8}), (3, {'a': 6, 'b': 9})]

    >>> import torch
    >>> list(recursive_zip(torch.tensor([1,2,3]), torch.tensor([4,5,6])))
    [(tensor(1), tensor(4)), (tensor(2), tensor(5)), (tensor(3), tensor(6))]
    """

    if len(args) == 1:
        if isinstance(args[0],(Sequence)):
            return args[0]
        if isinstance(args[0],torch.Tensor):
            return (x for x in args[0])
        if isinstance(args[0], dict):
            keys = args[0].keys()
            return (dict(zip(keys, vals)) for vals in zip(*(args[0][k] for k in keys)))
    return zip(*[recursive_zip(a) for a in args])


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=torch.float)
    elif isinstance(x, (int, float)):
        return torch.tensor(x, device=device, dtype=torch.float)
    elif isinstance(x, dict):
        return {k: to_tensor(v, device) for k,v in x.items()}
    elif isinstance(x, (list, tuple)):
        # If it's a list of dictionaries where each dictionary has the same keys, then convert it to a tensordict
        if all(isinstance(v, dict) for v in x) and all(set(x[0].keys()) == set(v.keys()) for v in x):
            keys = x[0].keys()
            return {k: torch.stack([to_tensor(v[k], device) for v in x], dim=0) for k in keys}
        return type(x)([to_tensor(v, device) for v in x])
    else:
        raise ValueError(f'Unknown type: {type(x)}')


def reset_hidden(terminal: torch.Tensor, hidden, initial_hidden, batch_dim):
    assert len(hidden) == len(initial_hidden)
    assert len(hidden) == len(batch_dim)
    output = tuple([
        torch.where(terminal.view(-1, *([1]*(len(h.shape)-d-1))), init_h, h)
        for init_h,h,d in zip(initial_hidden, hidden, batch_dim)
    ])
    #for h,ih,o,t,d in zip(hidden, initial_hidden, output, terminal, batch_dim):
    #    assert list(h.shape) == list(ih.shape)
    #    assert list(h.shape) == list(o.shape)
    return output


def action_dist_discrete(net_output, n=None):
    dist = torch.distributions.Categorical(logits=net_output['action'][:n])
    return dist, dist.log_prob


class TransformedTensor(torch.Tensor):
    # Save the untransformed value for log_prob. We run into numerical issues in SAC without this.
    untransformed_value: torch.Tensor


class SquashedGaussian(torch.distributions.Distribution):
    def __init__(self, loc, scale, low, high, log_prob_epsilon=1e-6):
        self.normal = torch.distributions.Normal(loc, scale)
        device = loc.device
        self._scale2 = torch.tensor((high - low) / 2, device=device)
        self._bias2 = torch.tensor((high + low) / 2, device=device)
        self._log_prob_epsilon = log_prob_epsilon

    def sample(self, sample_shape: _size = torch.Size()):
        x = self.normal.sample(sample_shape)
        s = self._scale2
        b = self._bias2
        output = torch.tanh(x) * s + b
        output = TransformedTensor(output)
        output.untransformed_value = x
        return output

    def rsample(self, sample_shape: _size = torch.Size()):
        x = self.normal.rsample(sample_shape)
        s = self._scale2
        b = self._bias2
        output = torch.tanh(x) * s + b
        output = TransformedTensor(output)
        output.untransformed_value = x
        return output

    def log_prob(self, value) -> torch.Tensor:
        if isinstance(value, TransformedTensor):
            untransformed_value = value.untransformed_value
        else:
            # Note: Applying the transformation and inverting it can lead to numerical issues. This path exists as a fallback, but it should be avoided if possible.
            untransformed_value = (torch.atanh((value - self._bias2) / self._scale2))
        log_prob = self.normal.log_prob(untransformed_value) - torch.log(self._scale2 * (1 - untransformed_value.tanh()**2) + self._log_prob_epsilon)
        return log_prob
    
    def entropy(self) -> torch.Tensor:
        # TODO
        return self.normal.entropy() * 0

    def __repr__(self):
        return f'SquashedGaussian(loc={self.normal.loc}, scale={self.normal.scale}, low={self._bias2 - self._scale2}, high={self._bias2 + self._scale2})'


def action_dist_continuous(net_output, n=None):
    action_mean = net_output['action_mean'][:n]
    action_logstd = net_output['action_logstd'][:n].clamp(-10, 10)
    dist = torch.distributions.Normal(action_mean, action_logstd.exp())
    return dist, lambda x: dist.log_prob(x).sum(-1)


def action_dist_continuous_squashed(net_output, n, low, high):
    action_mean = net_output['action_mean'][:n]
    action_logstd = net_output['action_logstd'][:n].clamp(-10, 10)
    dist = SquashedGaussian(action_mean, action_logstd.exp(), low, high)
    return dist, lambda x: dist.log_prob(x).sum(-1)


def action_dist_binomial(net_output, n=None):
    action_logit = net_output['action_logit'][:n]
    dist = torch.distributions.Binomial(logits=action_logit, total_count=1)
    return dist, lambda x: dist.log_prob(x).sum(-1)


def get_action_dist_function(action_space: gymnasium.Space, config={}):
    """
    Return a function that takes the network output and returns a distribution and a log probability function.

    Args:
        action_space: gymnasium.Space
        config: dict
            Used to determine the type of action distribution for different action spaces. For now, only continuous action spaces can be configured. If you want to use a squashed Gaussian distribution for continuous action spaces, set config = {'box': 'squashed'}. Otherwise, it will use a normal Gaussian distribution with clipped standard deviation by default.
    """
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return action_dist_discrete
    elif isinstance(action_space, gymnasium.spaces.Box):
        if config.get('box') == 'squashed':
            return lambda net_output, n=None: action_dist_continuous_squashed(net_output, n, action_space.low, action_space.high)
        else:
            return action_dist_continuous
    elif isinstance(action_space, gymnasium.spaces.MultiBinary):
        return action_dist_binomial
    else:
        raise NotImplementedError(f'Unknown action space: {action_space}')


def format_rate(num, unit_singular, unit_plural, time_diff):
    if time_diff == 0:
        return f'{num} {unit_plural if num != 1 else unit_singular} completed'
    if num > time_diff:
        return f'{num/time_diff:.2f} {unit_plural if num/time_diff != 1 else unit_singular}/s'
    else:
        return f'{time_diff/num:.2f} s/{unit_singular}'


# Models


class Model(torch.nn.Module):
    def forward(self, *inputs):
        raise NotImplementedError()


class FeedforwardModel(Model):
    ...


class RecurrentModel(Model):
    def __init__(self):
        super().__init__()

    def init_hidden(self, batch_size):
        raise NotImplementedError()

    @property
    def hidden_batch_dims(self):
        raise NotImplementedError()


class SequenceModel(Model):
    """ Model that accept a sequence of transitions.

    Example: decision transformers
    """
    ...
