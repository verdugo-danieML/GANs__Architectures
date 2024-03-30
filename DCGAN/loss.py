import torch
import torch.nn as nn

def generator_loss(fake_logits):
    """ Generator loss, takes the fake scores as inputs. """
    
    # Create labels for the fake images
    fake_labels = torch.ones_like(fake_logits)
    
    # Calculate the binary cross-entropy loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(fake_logits, fake_labels)
    
    return loss

def discriminator_loss(real_logits, fake_logits):
    """ Discriminator loss, takes the fake and real logits as inputs. """
    
    # Create labels for the real and fake images
    real_labels = torch.ones_like(real_logits)  # No label smoothing
    fake_labels = torch.zeros_like(fake_logits)
    
    # Calculate the binary cross-entropy loss for real and fake images
    criterion = nn.BCEWithLogitsLoss()
    real_loss = criterion(real_logits, real_labels)
    fake_loss = criterion(fake_logits, fake_labels)
    
    # Combine the losses
    loss = real_loss + fake_loss
    
    return loss