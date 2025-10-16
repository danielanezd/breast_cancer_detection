class EarlyStopping:
    def __init__(self, monitor="val_recall", patience=5, mode="max", delta=0.0):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            mode (str): 'min' or 'max' — minimize or maximize the monitored metric.
            delta (float): Minimum change to qualify as an improvement.
        """
        assert mode in ["min", "max"]
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor

    def step(self, current_value):
        if self.best_score is None:
            self.best_score = current_value
            return False

        improvement = (
            current_value < self.best_score - self.delta
            if self.mode == "min"
            else current_value > self.best_score + self.delta
        )

        if improvement:
            self.best_score = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
