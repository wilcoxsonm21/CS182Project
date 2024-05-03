#!/bin/bash

export UNAME=mortsven
export SCRATCH=/global/scratch/users/$UNAME

cd src

python3 ../generate_jobs_2.py --name more_batch1 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting18.yaml,conf/configs_3/prompting19.yaml,
python3 ../generate_jobs_2.py --name more_batch2 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting20.yaml,conf/configs_3/prompting21.yaml
python3 ../generate_jobs_2.py --name more_batch3 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting22.yaml,conf/configs_3/prompting23.yaml

cd sbatch

chmod +x more_batch1.sh
chmod +x more_batch2.sh
chmod +x more_batch3.sh

#sbatch batch1.sh
#sbatch batch2.sh
#sbatch batch3.sh
#sbatch batch4.sh
#sbatch batch5.sh
#sbatch batch6.sh
#sbatch batch7.sh
#sbatch batch8.sh
#sbatch batch9.sh
#sbatch batch10.sh
#sbatch batch11.sh
#sbatch batch12.sh