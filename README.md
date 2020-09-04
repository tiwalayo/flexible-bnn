# Flexible posteriors for Bayesian deep learning

This repo holds the code associated with the ideas I discuss in my post [Flexible posteriors for Bayesian deep learning](http://web.mit.edu/tiwa/www/musings/1-flexible.html). For the general framework, I extended [@martinferianc](https://github.com/martinferianc/BayesianNeuralNets)'s basic BNN architecture to accomodate the different types of networks I discuss in my post, using the same approaches outlined in the post. For the Kumaraswamy and NN-parameterized Bayesian networks, I use uniform priors.

## Structure

```
   .
   |-experiments            # Where all the experiments are located
   |---data                 # Where all the data is stored
   |---scripts              # Pre-configured scripts for running the experiments
   |-src                    # Main source folder, also containing the model descriptions
   |---models
   |-----pointwise
   |-----stochastic
   |-------bbb              # Implementation of Bayes-by-Backprop on MLP, Kumaraswamy nets, and Net-Nets.

```

## Experiments

There are in total three different experiments, regression, binary classification and MNIST digit classification. The default runners for the experiments are under the `experiments` folder, where the scripts for easy pre-configured runs can be found under `experiments/scripts/`. However, I only built the nets for regression, so your mileage may vary on the other types of experiments.

To run the experiments, simply prepare youself a a virtual environment and navigate to the `experiments/scripts` folder and pick one of the methods/tasks and run it as:

```
python3 bbb_binary_classification.py
```

No additional tuning should be necessary. However, the scripts assume that you have a GPU available for training and inference. In case you just wanto to use CPU, do:

```
python3 bbb_binary_classification.py --gpu -1
```


## Requirements

The main requirement is PyTorch>=1.5.0.

To be able to run this project and install the requirements simply execute (will work for GPUs or CPUs):

```
git clone https://github.com/martinferianc/BayesianNeuralNet-Tutorial
conda create --name venv --file requirements.txt --python=python3.6
conda activate venv
```
