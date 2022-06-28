from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

base_config = {
    "pct_complete": 'ep_pct_complete',
    "total_time": "ep_total_time",
    "total_distance": "ep_total_distance",
    "average_speed_kph": "ep_avg_speed",
    "average_displacement_error": "ep_avg_disp_err",
    "trajectory_efficiency":"ep_traj_efficiency",
    "trajectory_admissibility": "ep_traj_admissibility",
    "movement_smoothness": "movement_smoothness",
    "ep_n_steps":"ep_n_steps"
}


class TBLogger:
    """TensorBoard Logger instance."""

    def __init__(self,  log_dir, exp_name, config=base_config):
        """Initialize logger."""
        now = datetime.now()
        current_time = now.strftime("%m%d%H%M%S")
        log_dir = f"{log_dir}/tblogs/{exp_name}_{current_time}"

        self.writer = SummaryWriter(log_dir=log_dir)
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
            if local_metric_name in data:
                self.writer.add_scalar(tb_metric_name,
                                   data[local_metric_name], iter_no)
