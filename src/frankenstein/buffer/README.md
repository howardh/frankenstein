# Buffer

Utility classes that are useful for keeping track of transitions that have taken place.

We use two functions to log transitions: `append_obs()` and `append_action()`.

## Buffer Contents

- State
- Action
- Reward
- Terminal
- Misc

## Example usage

The following is an example environment loop from the Gymnasium documentation:
```python
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
```

Modified to save transitions to a `frankenstein.buffer.history.HistoryBuffer`:
```python
import gymnasium as gym
from frankenstein.buffer.history import HistoryBuffer

env = gym.make("LunarLander-v3", render_mode="human")
buffer = SimpleHistoryBuffer(max_len=1000) # Create a replay buffer that can hold 1000 transitions

observation, info = env.reset()
buffer.append_obs(observation) # <--

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    buffer.append_action(action) # <--

    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    buffer.append_obs(observation, reward, episode_over) # <--

env.close()
```

To sample from the replay buffer:
>>> buffer = SimpleHistoryBuffer(max_len=1000)
>>> buffer.append_obs(obs=1)
>>> buffer.append_action(1)
>>> buffer.append_obs(obs=2, reward=1, terminal=False)
>>> buffer.transition[0]
Transition(obs=1, action=1, next_obs=2, reward=1, terminal=False, misc=None, next_misc=None)



## Available Buffers

- `HistoryBuffer`
- `frankenstein.buffer.history.HistoryBuffer`
- `frankenstein.buffer.vec_history.VecHistoryBuffer`
