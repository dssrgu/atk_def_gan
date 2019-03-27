def get_sbatch_params(gpu_options):
    if gpu_options==1:
        nprocs=1
        cpus=10
        mem=30000
        gpus=1
    elif gpu_options==2:
        nprocs=2
        cpus=4
        mem=15000
        gpus=1
    elif gpu_options==3:
        nprocs=3
        cpus=2
        mem=10000
        gpus=1
    elif gpu_options==4:
        nprocs=4
        cpus=2
        mem=7500
        gpus=1
    elif gpu_options==5:
        nprocs=5
        cpus=2
        mem=6000
        gpus=1
    elif gpu_options==-1: # case where single task requires high cpu usage.
        nprocs=1
        cpus=20
        mem=60000
        gpus=2
    else:
        print("Invalid option")
        exit(0)
    return dict(nprocs=nprocs, cpus=cpus, mem=mem, gpus=gpus)

