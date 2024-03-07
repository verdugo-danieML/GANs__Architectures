import torch
import torch.nn as nn

# Calculate losses
def real_loss_vanilla(D_out, smooth=False):
    # label smoothing
    labels = torch.ones_like(D_out) * (0.9 if smooth else 1.0)
    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out, labels)
    return loss

def fake_loss_vanilla(D_out):
    labels = torch.zeros_like(D_out) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out, labels)
    return loss