import xgboost as xgb
from tensorboardX import SummaryWriter


class TBCallback(xgb.callback.TrainingCallback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def after_iteration(self, model, epoch, evals_log):
        for data, metrics in evals_log.items():
            for metric, values in metrics.items():
                self.writer.add_scalar(f"{data}/{metric}", values[-1], epoch)
        return False
