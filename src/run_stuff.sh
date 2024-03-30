eval "$(/home/ubuntu/mydata/miniconda3/bin/conda shell.bash hook)"
conda activate in-context-learning2
python3 train.py --config conf/special_tests/big_prompting_degree_5.yaml
python3 train.py --config conf/special_tests/lora_full_degree_5.yaml
python3 train.py --config conf/special_tests/lora_full_shared_roots.yaml
python3 train.py --config conf/special_tests/lora_small_full_degree_5.yaml
python3 train.py --config conf/special_tests/normal_prompting_degree_5.yaml
python3 train.py --config conf/special_tests/normal_prompting_shared_roots.yaml