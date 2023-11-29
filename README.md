This repository contains the code and models for our CS182 Fall 2023 Project:
## **Polynomial Regression Using In-Context Learning with Transformer Models**
Ria Doshi*, Stefanie Gschwind*, Carolyn Wang*, Max Wilcoxson* <br>


## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download model checkpoints from the link provided in `example_checkpoint.txt` to `../models/kernel_linear_regression`

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `test.ipynb` notebook contains code to load our own pre-trained model from the curriculum learning experiment and plot the model performance. You can toggle the noise flag to see how the model performs on noisy data. 
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy_chebyshev.yaml` for a training run. This will take about 6-7 hours to run on a single NVIDIA 3090. 

Codebase Forked From the Below Paper: <br>
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>
