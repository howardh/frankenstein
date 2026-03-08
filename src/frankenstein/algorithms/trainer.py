import os
import datetime
import time

import torch


class Trainer:
    def __init__(self, env, model, optimizer, scheduler, device):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.transition_count = 0
        self.gradient_step_count = 0

    def train(self):
        ...

    def train_gradient_step(self, update_model: bool = True, return_gradients: bool = False):
        """ Perform one gradient step of training. """
        ...

    def train_environment_step(self, update_model: bool = True, return_gradients: bool = False):
        """ Perform one environment step of training. """
        ...


class Checkpoint:
    def __init__(self, data: dict, frequency: tuple, path: str):
        """
        Args:
            data (dict): Data to save in the checkpoint. Values should have a `state_dict` method and a `load_state_dict` method (e.g. torch modules and optimizers).
            frequency (int): The number of iterations between each checkpoint.
            path (str): Path to save the checkpoint. If the path does not exist, it will be created.
        """
        self.data = data
        self.frequency = frequency
        self.path = path

        for key, value in self.data.items():
            if not hasattr(value, "state_dict") or not hasattr(value, "load_state_dict"):
                raise ValueError(f"Value for key {key} does not have 'state_dict' and 'load_state_dict' methods.")

        self.start_step = 0

        self._last_checkpoint_step = 0
        self._last_checkpoint_time = 0

        self.load()

    def _should_save(self, step: int):
        match self.frequency:
            case (freq, 'steps') | (freq, 'step'):
                return (step - self._last_checkpoint_step) >= freq
            case (freq, 'seconds') | (freq, 'second'):
                return time.time() - self._last_checkpoint_time >= freq
            case (freq, 'minutes') | (freq, 'minute'):
                return (time.time() - self._last_checkpoint_time) / 60 >= freq
            case (freq, 'hours') | (freq, 'hour'):
                return (time.time() - self._last_checkpoint_time) / 3600 >= freq
            case _:
                raise ValueError(f"Invalid frequency: {self.frequency}")

    def save(self, step: int, force: bool = False):
        """
        Save the checkpoint to disk.

        Args:
            step (int): The current step.
            force (bool): If True, save the checkpoint regardless of what step it is.
        """
        if not force:
            if not self._should_save(step):
                return
            if step < self.start_step:
                raise ValueError(f"Step {step} is less than the start step {self.start_step}.")
            if step == self.start_step:
                return # We either just loaded the checkpoint or haven't started training yet. Don't waste time saving it again.

        # Save the checkpoint
        # Save it to a temporary file first, then rename it to the final file name so that we don't end up with a corrupted file if the process is interrupted.
        checkpoint = {key: value.state_dict() for key, value in self.data.items()}
        checkpoint["step"] = step
        filename = os.path.join(self.path, f"checkpoint.pt")
        temp_filename = filename + ".temp"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(checkpoint, temp_filename)
        os.rename(temp_filename, filename)
        print(f"Saved checkpoint to {os.path.abspath(filename)}")

        self._last_checkpoint_step = step
        self._last_checkpoint_time = time.time()
        print(f"  Step: {step}")
        print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def load(self):
        file_path = os.path.join(self.path, "checkpoint.pt")
        if not os.path.exists(file_path):
            print(f"No checkpoint found at {os.path.abspath(file_path)}")
            self._last_checkpoint_time = time.time()
            return
        print(f"Loading checkpoint from {os.path.abspath(file_path)}")
        checkpoint = torch.load(file_path)

        # Validation
        if len(checkpoint) != len(self.data) + 1:
            raise ValueError(f"Checkpoint data does not match expected data. Checkpoint file contains {len(checkpoint)} entries ({checkpoint.keys()}). Expected {len(self.data)} entries (self.data.keys()) plus 'step'.")

        # Load the data
        for key in self.data:
            self.data[key].load_state_dict(checkpoint[key])
        self.start_step = checkpoint["step"]

        self._last_checkpoint_step = self.start_step
        self._last_checkpoint_time = time.time()


class NullCheckpoint(Checkpoint):
    def __init__(self):
        super().__init__({}, 1, "")

    def save(self, *args, **kwargs):
        pass

    def load(self):
        pass
