fork of https://github.com/kuangliu/pytorch-cifar
Files relevant to the project:

main.py:

code for handling training, selective regularization, and selective model saving. Modified to reduce learning rate on a predefined schedule automatically instead of relying on manual restarts with smaller learning rates.

models/simple_densenet.py

an implementation of DenseNet with modifications to simplify, as well as to support selective regularization and saving.

project.ipynb:

jupyter notebook version of visualize.py

visualize.py:

older visualization code before I converted it to a jupyter notebook.

All other files are unmodified from the original forked repository, and are not used for our experiments.

