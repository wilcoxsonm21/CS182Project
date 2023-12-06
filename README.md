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

2. To train a Polynomial Regression Model make sure you are on the branch 'sanity-checks' and run the following line in your terminal: 

    ```
    python train.py --config conf/toy_chebyshev.yaml
    ```

    Edit the toy_chebyshev.yaml file you want to change aspects of the model or training such as: batch size, learning rate, # training steps, the range of polynomial degrees the model is trained on, or other aspects of the curriculum. Two brief notes on changing the range of the polynomial degrees for training:
        i. To change the lowest degree polynomial seen, change the "lowest_degree" value inside of the 'task_kwargs' dictionary
        ii. To change the highest degree polynomial seen, chnage BOTH the 'start' and 'end' parameters found under 'deg' to the degree value you want it to be

    To change whether or not you have noise in your training you need to edit the tasks.py file in two locations. Inside of the tasks.py file, change the 'noise' flag in the evaluate function definition (line 159) to 'True' and set the noise_variance to be your desired value. Also in tasks.py file in the init of NoisyLinearRegression, set the 'noise_std' to your desired value.
    
3. To evaluate your model:

Codebase Forked From the Below Paper: <br>
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>
