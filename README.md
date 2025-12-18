# Dynamical-Variational-Autoencoders

This project explores the application of Variational Autoencoders (VAEs) to sequential data, such as time series or audio. Applying VAEs to sequential data presents a specific challenge: standard VAEs assume independent data points, whereas sequential data has strong temporal dependencies. To address this, Recurrent Neural Networks (RNNs) must be integrated into the VAE architecture to capture temporal context. 

## Project Overview

We replicate and experiment with two specific architectures presented in the paper *A Recurrent Latent Variable Model for Sequential Data* (arXiv:2008.12595). :

1. VRNN (Variational Recurrent Neural Network): A model where the VAE is applied at every time step, conditioned on the RNN state.
2. SRNN (Stochastic Recurrent Neural Network): A hierarchical architecture designed to better separate deterministic and stochastic information.

## Results
The comprehensive results of our experiments are available in the [Experiments with VRNN.ipynb] notebook within the "experiments" folder.

## Dataset

We conducted our experiments on a subset of the VCTK corpus, a speech dataset containing audio recordings from various English speakers. For this project, the raw waveform data (.wav) was converted into Mel-spectrograms, a standard time-frequency representation for audio processing.

## Objectives

The primary goals of this implementation were to:

* **Re-implement Architectures:** Build the VRNN and SRNN architectures from scratch using PyTorch.
* **Hyperparameter Optimization:** Train models by selecting optimal parameters, with a specific focus on:
* Choice of prior distribution (Gaussian vs. Student-t).
* Duration of the Kullback-Leibler (KL) divergence annealing phase.
* Number of epochs and learning rate scheduling.


* **Reconstruction Evaluation:** Assess the quality of audio reconstruction across the different architectures and priors.
* **Audio Generation:** Experiment with generating audio in two modes:
* **Cold Start:** Generating from scratch with no initial context.
* **Priming:** Using a "warm-up" phase where the model is fed a short sequence of real data to initialize the hidden state.


* **Refinement:** Analyze and refine the models and generation process to improve the intelligibility of the output audio.



