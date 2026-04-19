class BaseEnvironment:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError