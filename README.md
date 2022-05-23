# TBPTT

Some implementations of truncated back-propagation through time (TBPTT) in Pytorch.

1. `demo_15500.py`
    - from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500
    - problem: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`
2. `demo_lightning.py`
    - from https://github.com/PyTorchLightning/pytorch-lightning/issues/293
    - problem: `only support k1 == k2 == tbptt_steps`

![tbptt](tbtpp.gif)

TBPTT with k1=2, k2=5. 


This is a special case of TBPTT where you backpropagate k1 losses at every k1 steps back for k2 steps. In the method described in Sutskever's Thesis, you only back propagate from t down to k2 whenever t divides by k1. Here we backpropagate t, t-1,t-2,t-3...,t-k1. This case of TBPTT is useful when training sequences are too long for LSTM to handle or are too computationally expensive to train, but good predictions are required for all targets. 

Based on the code from https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500 

related resources:
1. https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf (see page 23)

