import getpass
import os
import subprocess

from autopar import make_par
from sbatch_launcher import launch_tasks, srun_gpuless_task

PARTITION = 'all'  # 'mllab', 'all', or 'dept'

TENSORBOARD_DIR = 'mnist/data'

srun_gpuless_task(r"""bash -c 'tensorboard --host=$(hostname).mllab.snu.ac.kr --port=0 --logdir={}'""".format(TENSORBOARD_DIR))

