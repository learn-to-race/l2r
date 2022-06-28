import os
import sys
import logging
import re
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
import shutil
import warnings

base_config = {
    "pct_complete": 'ep_pct_complete',
    "total_time": "ep_total_time",
    "total_distance": "ep_total_distance",
    "average_speed_kph": "ep_avg_speed",
    "average_displacement_error": "ep_avg_disp_err",
    "trajectory_efficiency":"ep_traj_efficiency",
    "trajectory_admissibility": "ep_traj_admissibility",
    "movement_smoothness": "movement_smoothness",
    "ep_n_steps":"ep_n_steps",
    "episodic_return":"episodic_return",
    "ep_interventions":"ep_interventions"
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



def setup_tb_logging(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')

    return SummaryWriter(log_dir=log_dir)


def setup_file_logging(logdir, experiment_name):
    # write experimental config to logfile
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{logdir}/runlogs/{experiment_name}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.info


def setup_logging(logdir, experiment_name, resume):

    if not os.path.exists(f'{logdir}/runlogs'):
        os.umask(0)
        os.makedirs(logdir, mode=0o777, exist_ok=True)
        os.makedirs(f"{logdir}/runlogs", mode=0o777, exist_ok=True)
        os.makedirs(f"{logdir}/tblogs", mode=0o777, exist_ok=True)

    file_logger = setup_file_logging(logdir, experiment_name)
    tb_logger = setup_tb_logging(logdir, experiment_name, resume)
    return (file_logger, tb_logger)


def find_envvar_patterns(config, key):
    pattern = re.compile(r'.*?\${(\w+)}.*?')
    try:
        envvars = re.findall(pattern, config[key])
    except BaseException:
        envvars = []
        pass
    return envvars


def replace_envvar_patterns(config, key, envvars, args):
    for i, var in enumerate(envvars):
        if var == "DIRHASH":
            dirhash = '{}/'.format(
                args.dirhash) if not args.runtime == 'local' else ''
            config[key] = config[key].replace('${' + var + '}', dirhash)
        if var == "PREFIX":
            prefix = {'local': '/data', 'phoebe': '/mnt'}
            config[key] = config[key].replace(
                '${' + var + '}', prefix[args.runtime])
        else:
            config[key] = config[key].replace(
                '${' + var + '}', os.environ.get(var, var))


def resolve_envvars(config, args):

    for key in list(config.keys()):

        if isinstance(config[key], dict):
            # second level
            for sub_key in list(config[key].keys()):
                sub_envvars = find_envvar_patterns(config[key], sub_key)
                if len(sub_envvars) > 0:
                    for _ in sub_envvars:
                        replace_envvar_patterns(
                            config[key], sub_key, sub_envvars, args)

        envvars = find_envvar_patterns(config, key)
        if len(envvars) > 0:
            replace_envvar_patterns(config, key, envvars, args)

    return config





class RecordExperience:

    def __init__(
            self,
            record_dir,
            track,
            experiment_name,
            logger,
            agent=False):

        self.record_dir = record_dir
        self.track = track
        self.experiment_name = experiment_name
        self.filename = 'transition'
        self.agent = agent
        self.logger = logger

        self.path = os.path.join(
            self.record_dir,
            self.track,
            self.experiment_name)

        self.logger('Recording agent experience')

    def save(self, record):

        filename = f"{self.path}/{record['stage']}/{record['episode']}/{self.filename}_{self.experiment_name}_{record['step']}"

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(
            os.path.join(
                self.path, record['stage'], str(
                    record['episode'])), exist_ok=True)

        np.savez_compressed(filename, **record)

        return record

    def save_thread(self):
        """Meant to be run as a separate thread
        """
        if not self.agent:
            raise Exception('RecordExperience requires an SACAgent')

        while True:
            batch = self.agent.save_queue.get()
            self.logger('[RecordExperience] Saving experience.')
            for record in batch:
                self.save(record)
