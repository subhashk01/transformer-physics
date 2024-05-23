
# How Do Transformers "Do" Physics?     

This is the github repo for the paper 'How Do Transformers "Do" Physics? Investigating the Simple Harmonic Oscillator'. INSERT PAPER. 

How do transformers model physics? By investigating model internals, we find that transformers likely use the matrix exponential numerical method to model the simple harmonic oscillator. We believe our methodology will also help uncover that transformers use to model higher order linear differential equations and some nonlinear systems, potentially revealing their internal "world model."

## How to use this repo
All paper results are directly reproducible from this repo, but because I'm graduating I'm a bit pressed for time to make this repo usable. Here's a brief walk through of the file system, with the promise of making the code more readable in the future. Reach out to me directly if you're interested in this implementation.

* **data.py** - Used to generate all data for linear regression, undamped, and damped harmonic oscillators used to train our transformers
* **model.py** - We define our model architecture, based on [Karpathy's minGPT implementation](\href{https://github.com/karpathy/minGPT/tree/master}), with support model interventions.
* **train_models.py** - Exactly what it sounds like
* **analyze_models.py** - Used to generate Figs 10,13
* **generate_probe_targets.py** - We create probe targets for the intermediates of the numerical methods we hypothesize transformers use
* **train_probes.py** - We then probe the model at all internal locations for probe targets
* **analyze_probes.ipynb** - All other figures in the paper can be reconstructed from this notebook.

Cheers!

