#!/bin/bash

cd src

python3 ../generate_jobs_2.py --name batch1 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting1.yaml,conf/configs_2/prompting2.yaml,conf/configs_2/prompting3.yaml,conf/configs_2/prompting4.yaml,
python3 ../generate_jobs_2.py --name batch2 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting5.yaml,conf/configs_2/prompting6.yaml,conf/configs_2/prompting7.yaml,conf/configs_2/prompting8.yaml,
python3 ../generate_jobs_2.py --name batch3 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting9.yaml,conf/configs_2/prompting10.yaml,conf/configs_2/prompting11.yaml,conf/configs_2/prompting12.yaml,
python3 ../generate_jobs_2.py --name batch4 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting13.yaml,conf/configs_2/prompting14.yaml,conf/configs_2/prompting15.yaml,conf/configs_2/prompting16.yaml
python3 ../generate_jobs_2.py --name batch5 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting17.yaml,conf/configs_2/prompting18.yaml,conf/configs_2/prompting19.yaml,conf/configs_2/prompting20.yaml,
python3 ../generate_jobs_2.py --name batch6 --conda_env_name in-context-learning3 python train.py --config conf/configs_2/prompting21.yaml,conf/configs_2/prompting22.yaml,conf/configs_2/prompting23.yaml,conf/configs_3/prompting1.yaml,
python3 ../generate_jobs_2.py --name batch7 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting2.yaml,conf/configs_3/prompting3.yaml,conf/configs_3/prompting4.yaml,conf/configs_3/prompting5.yaml,
python3 ../generate_jobs_2.py --name batch8 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting6.yaml,conf/configs_3/prompting7.yaml,conf/configs_3/prompting8.yaml,conf/configs_3/prompting9.yaml
python3 ../generate_jobs_2.py --name batch9 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting10.yaml,conf/configs_3/prompting11.yaml,conf/configs_3/prompting12.yaml,conf/configs_3/prompting13.yaml,
python3 ../generate_jobs_2.py --name batch10 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting14.yaml,conf/configs_3/prompting15.yaml,conf/configs_3/prompting16.yaml,conf/configs_3/prompting17.yaml,
python3 ../generate_jobs_2.py --name batch11 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting18.yaml,conf/configs_3/prompting19.yaml,conf/configs_3/prompting20.yaml,conf/configs_3/prompting21.yaml,
python3 ../generate_jobs_2.py --name batch12 --conda_env_name in-context-learning3 python train.py --config conf/configs_3/prompting22.yaml,conf/configs_3/prompting23.yaml,conf/configs_4/p100degree.yaml,conf/configs_4/p100shared.yaml

cd sbatch

chmod +x batch1.sh
chmod +x batch2.sh
chmod +x batch3.sh
chmod +x batch4.sh
chmod +x batch5.sh
chmod +x batch6.sh
chmod +x batch7.sh
chmod +x batch8.sh
chmod +x batch9.sh
chmod +x batch10.sh
chmod +x batch11.sh
chmod +x batch12.sh

sbatch batch1.sh
sbatch batch2.sh
sbatch batch3.sh
sbatch batch4.sh
sbatch batch5.sh
sbatch batch6.sh
sbatch batch7.sh
sbatch batch8.sh
sbatch batch9.sh
sbatch batch10.sh
sbatch batch11.sh
sbatch batch12.sh