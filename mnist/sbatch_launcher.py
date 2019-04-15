import itertools
import os
import signal
import subprocess

from sbatch_params import get_sbatch_params

def launch_tasks(num_procs_on_gpu,
                 base_cmd,
                 param_dict,
                 partition='mllab',
                 qos='normal',
                 timeout='48:00:00',
                 sbatch_additional_params=''):
    sbp_dict = get_sbatch_params(num_procs_on_gpu)
    nprocs = sbp_dict['nprocs']
    cpus = sbp_dict['cpus']
    mem = sbp_dict['mem']
    gpus = sbp_dict['gpus']

    param_keys = [str(v) for v in param_dict.keys()]
    nkey = len(param_keys)
    param_list = [v for v in itertools.product(*tuple([param_dict[key] for key in param_keys]))]

    for i in range(0,len(param_list), nprocs): 
        cmd_pair = ""
        for j in range(nprocs):
            if (i+j >= len(param_list)):
                    break
            param = param_list[i+j]
            cmd = base_cmd + ' ' + ''.join([
                '{} {} '.format(param_keys[key_idx], param[key_idx])
                for key_idx in range(nkey)])
            cmd_pair += "'{}'".format(cmd) + " "
        sbatch_cmd = "sbatch --partition={} --qos={} --time={} {} --ntasks={} --cpus-per-task={} --mem={} --gres=gpu:{} ./run_general.sh {}".format(
                partition, qos, timeout, sbatch_additional_params, nprocs, cpus, mem, gpus, cmd_pair)
        print(sbatch_cmd)
        subprocess.check_call(sbatch_cmd, shell=True)

def srun_gpuless_task(cmd,
                      cpus=4,
                      mem=15000,
                      partition='mllab',
                      qos='normal',
                      timeout='48:00:00'):
    srun_cmd = r'''srun --partition={} --qos={} --time={} --ntasks=1 --cpus-per-task={} --mem={} {}'''.format(
            partition, qos, timeout, cpus, mem, cmd)
    print(srun_cmd)
    p = subprocess.Popen(srun_cmd, shell=True)
    srun_pid = p.pid
    def handler(signum, frame):
        os.kill(srun_pid, signum)
    signal.signal(signal.SIGINT, handler)
    p.wait()

