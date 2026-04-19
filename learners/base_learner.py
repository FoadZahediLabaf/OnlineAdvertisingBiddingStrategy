class BaseLearner:
    def select_action(self):
        raise NotImplementedError

    def update(self, feedback):
        raise NotImplementedError
    