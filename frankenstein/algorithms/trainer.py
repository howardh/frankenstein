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
