import torch
import torch.nn as nn

"""
from: https://github.com/PyTorchLightning/pytorch-lightning/issues/293

only support k1 == k2 == tbptt_steps
"""

n_features = 1
out_size = 2
tbptt_steps = 3

seq_len_1 = 4
seq_len_2 = 5

input_is_packed = True
batch_first = True
split_dim = 1 if batch_first else 0

# Construct the input. x and y are represent one batch (i.e. `x, y = batch`)
if input_is_packed:
    x = [torch.rand(seq_len_1, n_features), torch.rand(seq_len_2, n_features)]
    y = [torch.rand(seq_len_1, out_size), torch.rand(seq_len_2, out_size)]
    # Pack x and target y into a PackedSequence, that can handle variable input sizes
    x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    y = nn.utils.rnn.pack_sequence(y, enforce_sorted=False)
else:
    # Sequences in one batch have to have the same length
    x = torch.rand(seq_len_1, 2, n_features)
    y = torch.rand(seq_len_1, 2, out_size)
    if batch_first:
        x.transpose_(0, 1)
        y.transpose_(0, 1)

# The dataloader needs to output a batch like
# x, y = batch

# The following part would need to be implemented in Trainer / LightningModule

model = nn.RNN(
    input_size=n_features, hidden_size=out_size, num_layers=2, batch_first=batch_first
)

opt = torch.optim.Adam(model.parameters())

# In the begining of each batch
if input_is_packed:
    x, x_lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=batch_first)
    y, y_lengths = nn.utils.rnn.pad_packed_sequence(y, batch_first=batch_first)

# Iterate over the sequence in tbptt_steps in each batch
state = None
for i, (x_, y_) in enumerate(
    zip(x.split(tbptt_steps, dim=split_dim), y.split(tbptt_steps, dim=split_dim))
):
    print(f"{i} x: {x_.shape}")
    print(f"{i} y: {y_.shape}")
    opt.zero_grad()
    model.train()
    # Detach last hidden state, so the backprop-graph will be cut
    if state is not None:
        state.detach_()
    # Forward path
    y_pred, state = model(x_, state)
    # Compute loss
    loss = nn.functional.mse_loss(y_, y_pred)
    # Backward path
    loss.backward()
    # Update weights
    opt.step()
