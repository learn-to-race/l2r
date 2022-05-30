from torch.utils.tensorboard import SummaryWriter

base_config = {
    "pct_complete": 'val/ep_pct_complete',
    "total_time": "val/ep_total_time",
    "total_distance": "val/ep_total_distance",
    "average_speed_kph": "val/ep_avg_speed",
    "average_displacement_error":"val/ep_avg_disp_err",
    "trajectory_admissibility":"val/ep_traj_admissibility",
    "movement_smoothness":"val/movement_smoothness"
}


class TBLogger:
    """TensorBoard Logger instance."""

    def __init__(self, config=base_config):
        """Initialize logger."""
        self.writer = SummaryWriter()
        self.config = config

    def log(self, data, iter_no):
        """Send metric information to TensorBoard

        Args:
            data (dict): Dict of local metric name to metric value
            iter_no (int): Iteration number
        """

        # If there is no data, don't log anything.
        if data == dict():
            return

        for local_metric_name, tb_metric_name in self.config:
            self.writer.add_scalar(tb_metric_name,
                                   data[local_metric_name], iter_no)