import getpass
import os
import subprocess

from autopar import make_par
from sbatch_launcher import launch_tasks, srun_gpuless_task

NUM_PROCS_ON_GPU = 1
PARTITION = 'all'  # 'mllab', 'all', or 'dept'

PYTHON_FILE = 'main.py'
#PYTHON_FILE = 'test_adversarial_examples.py'
IMPORT_PATHS = ['.']
TENSORBOARD_DIR = 'mnist/data'
COMMON_PARAMS = '--log_base_dir {}'.format(TENSORBOARD_DIR)
PYTHON_VERSION = 'PY3'  # 'PY2' or 'PY3'
LABEL = 'test_exp'
QOS_TYPE = 'normal'  # 'normal' or 'highprio'

PARAM_DICT = {
    '--seeds': [i for i in range(1)],
    '--E_lr': [0.01, 0.001],
    '--defG_lr': [0.1, 0.01],
    '--advG_lr': [0.001, 0.0001],
    '--mine_lr': [0.001, 0.0001, 0.00001],
    '--overwrite': ['False'],
    '--logging': ['True'],
}

# If you don't use python, you can just assign the base command string to
# BASE_CMD directly, like BASE_CMD = '/.my_binary --foo bar'
BASE_CMD = './{} {}'.format(
        make_par(main_file=PYTHON_FILE,
                 python_version=PYTHON_VERSION,
                 label=LABEL,
                 import_paths=IMPORT_PATHS),
        COMMON_PARAMS)

# Setting environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64/:/usr/local/cudnns/cudnn_7.3.1/lib64/:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LIBRARY_PATH'] = '/usr/local/cuda-9.0/lib64/:/usr/local/cudnns/cudnn_7.3.1/lib64/:' + os.environ.get('LIBRARY_PATH', '')
os.environ['CPATH'] = '/usr/local/cuda-9.0/include/:/usr/local/cudnns/cudnn_7.3.1/include/:' + os.environ.get('CPATH', '')
os.environ['LD_LIBRARY_PATH'] = r'/usr/local/cuda-9.0/extras/CUPTI/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

for key in ['LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH']:
    if os.environ[key][-1] == ':':
        os.environ[key] = os.environ[key][:-1]

assert QOS_TYPE in ['normal', 'highprio']
if QOS_TYPE == 'highprio':
    QOS_TYPE += '-{}'.format(getpass.getuser())

# Setting up TENSORBOARD_DIR
if os.path.exists(TENSORBOARD_DIR):
    tb_dir_real = TENSORBOARD_DIR
    if os.path.islink(tb_dir_real):
        tb_dir_real = os.readlink(tb_dir_real)
    tb_dir_real = os.path.abspath(tb_dir_real)
    if tb_dir_real != '/data/busy_update_store' and not tb_dir_real.startswith('/data/busy_update_store/'):
        print("\033[91mExisting TENSORBOARD_DIR('{}') is not under '/data/busy_update_store', so tensorboard update will be VERY slow (every 1800s).\033[0m".format(TENSORBOARD_DIR))
else:
    tb_dir_real = os.path.join('/data/busy_update_store', getpass.getuser(), TENSORBOARD_DIR)
    if not os.path.exists(tb_dir_real):
        os.makedirs(tb_dir_real)
    cwd = os.path.abspath(os.getcwd())
    tb_dir_parent, tb_dir_name = os.path.split(TENSORBOARD_DIR)
    if tb_dir_parent != '':
        os.chdir(tb_dir_parent)
    os.symlink(tb_dir_real, tb_dir_name)
    os.chdir(cwd)

launch_tasks(
        num_procs_on_gpu=NUM_PROCS_ON_GPU,
        base_cmd=BASE_CMD,
        param_dict=PARAM_DICT,
        partition=PARTITION,
        qos=QOS_TYPE,
)

srun_gpuless_task(r"""bash -c 'tensorboard --host=$(hostname).mllab.snu.ac.kr --port=0 --logdir={}'""".format(TENSORBOARD_DIR))

