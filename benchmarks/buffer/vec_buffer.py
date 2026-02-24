import timeit
from pprint import pprint
import numpy as np

from frankenstein.buffer.vec_history import ListBackedVecHistoryBuffer, NumpyBackedVecHistoryBuffer, DataSizes

if __name__ == '__main__':
    def init_buffer(buffer, n):
        num_envs = 16
        obs = np.random.rand(num_envs, 3, 3, 3).astype(np.float32)
        buffer.append_obs(obs)
        action = np.random.rand(num_envs, 2).astype(np.float32)
        buffer.append_action(action)
        for _ in range(n):
            obs = np.random.rand(num_envs, 3, 3, 3).astype(np.float32)
            reward = np.random.rand(num_envs).astype(np.float32)
            terminated = (np.random.rand(num_envs) < 0.1)
            truncated = (np.random.rand(num_envs) < 0.1)
            misc = None
            action = np.random.rand(num_envs, 2).astype(np.float32)

            buffer.append_obs(obs, reward, terminated, truncated, misc)
            buffer.append_action(action)

    def benchmark_append(buffer, num_envs=16, num_rows=20_000):
        obs = np.random.rand(num_envs, 3, 3, 3).astype(np.float32)
        action = np.random.rand(num_envs, 2).astype(np.float32)
        reward = np.random.rand(num_envs).astype(np.float32)
        terminated = (np.random.rand(num_envs) < 0.1)
        truncated = (np.random.rand(num_envs) < 0.1)
        misc = None

        buffer.append_obs(obs)
        buffer.append_action(action)
        for _ in range(num_rows):
            buffer.append_obs(obs, reward, terminated, truncated, misc)
            buffer.append_action(action)

    def benchmark_transitions(buffer):
        for i in range(buffer.num_transitions):
            buffer.get_transition(i)

    def benchmark_trajectories(buffer):
        for i in range(buffer.num_trajectories):
            buffer.get_trajectory(i)

    def make_buffers(max_len, num_envs, trajectory_length, num_rows):
        buffer = NumpyBackedVecHistoryBuffer(
            max_len=max_len,
            num_envs=num_envs,
            trajectory_length=trajectory_length,
            #data_size=DataSizes(obs=4*3*3*3, reward=4, misc=0, action=4*2),
        )
        buffer_ref = ListBackedVecHistoryBuffer(
            max_len=max_len,
            num_envs=num_envs,
            trajectory_length=trajectory_length,
            #device=torch.device('cpu'),
        )
        init_buffer(buffer, num_rows)
        init_buffer(buffer_ref, num_rows)
        return buffer, buffer_ref

    benchmark_funcs = {
        'append': benchmark_append,
        'get_transition': benchmark_transitions,
        'get_trajectory': benchmark_trajectories,
    }
    scale = { # Rescale the benchmark times to make them more comparable
        'append': lambda _: 20_000,
        'get_transition': lambda buffer: buffer.num_transitions,
        'get_trajectory': lambda buffer: buffer.num_trajectories,
    }

    # An arbitrary configuration
    for name, func in benchmark_funcs.items():
        buffer, buffer_ref = make_buffers(
            max_len=210,
            num_envs=16,
            trajectory_length=200,
            num_rows=300,
        )

        number = 10
        print(name, func)
        print('\t', 'Benchmarking list implementation ...')
        print('\t  ', timeit.timeit(lambda: func(buffer_ref), number=number))
        print('\t', 'Benchmarking numpy implementation ...')
        print('\t  ', timeit.timeit(lambda: func(buffer), number=number))

    """
    append <function benchmark_append at 0x7fb3cb74c4a0>
             Benchmarking list implementation ...
               2.3130773000011686
             Benchmarking numpy implementation ...
               1.73847976300749
    get_transition <function benchmark_transitions at 0x7fb3cb74c360>
             Benchmarking list implementation ...
               1.7275167650077492
             Benchmarking numpy implementation ...
               1.619693862987333
    get_trajectory <function benchmark_trajectories at 0x7fb370f33420>
             Benchmarking list implementation ...
               3.042548962010187
             Benchmarking numpy implementation ...
               0.07056194100005087
    """

    ## Test different trajectory lengths
    #print()
    #print('==================================================')
    #print('Testing different trajectory lengths')
    #print()
    #results_by_trajectory_length = {}
    #for traj_len in [10, 50, 100, 200]:
    #    for name, func in benchmark_funcs.items():
    #        buffer_np, buffer_list = make_buffers(
    #            max_len=210,
    #            num_envs=16,
    #            trajectory_length=traj_len,
    #            num_rows=300,
    #        )

    #        print(name, func)
    #        print('\t', 'Benchmarking list implementation ...')
    #        results_list = timeit.timeit(lambda: func(buffer_list), number=100) / scale[name](buffer_list)
    #        print('\t  ', results_list)
    #        print('\t', 'Benchmarking numpy implementation ...')
    #        results_np = timeit.timeit(lambda: func(buffer_np), number=100) / scale[name](buffer_np)
    #        print('\t  ', results_np)

    #        results_by_trajectory_length[(name, 'list', traj_len)] = results_list
    #        results_by_trajectory_length[(name, 'numpy', traj_len)] = results_np

    #pprint(results_by_trajectory_length)

    ## Test different numbers of environments
    #print()
    #print('==================================================')
    #print('Testing different numbers of environments')
    #print()
    #results_by_num_envs = {}
    #for num_envs in [1, 4, 16, 64]:
    #    for name, func in benchmark_funcs.items():
    #        buffer, buffer_ref = make_buffers(
    #            max_len=210,
    #            num_envs=num_envs,
    #            trajectory_length=200,
    #            num_rows=300,
    #        )

    #        print(name, func)
    #        print('\t', 'Benchmarking list implementation ...')
    #        results_list = timeit.timeit(lambda: func(buffer_ref), number=100) / scale[name](buffer_ref)
    #        print('\t  ', results_list)
    #        print('\t', 'Benchmarking numpy implementation ...')
    #        results_np = timeit.timeit(lambda: func(buffer), number=100) / scale[name](buffer)
    #        print('\t  ', results_np)

    #        results_by_num_envs[(name, 'list', num_envs)] = results_list
    #        results_by_num_envs[(name, 'numpy', num_envs)] = results_np

    #pprint(results_by_trajectory_length)
