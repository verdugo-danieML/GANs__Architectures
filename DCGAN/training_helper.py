import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from loss import generator_loss, discriminator_loss
import pickle as pkl

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(14, 4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) / 2 * 255).astype(np.uint8)  # Rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img)
    plt.show()
    
def generator_step(generator, discriminator, g_optimizer, batch_size, latent_dim, device):
    """ One training step of the generator. """
    
    # Clear the gradients
    g_optimizer.zero_grad()
    
    # Generate fake images
    noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)  # Move noise to the same device as the model
    fake_images = generator(noise)
    
    # Get the discriminator's predictions for the fake images
    fake_logits = discriminator(fake_images)
    
    # Calculate the generator loss
    g_loss = generator_loss(fake_logits)
    
    # Backpropagate the gradients
    g_loss.backward()
    
    # Update the generator's parameters
    g_optimizer.step()
    
    return g_loss.item()

def discriminator_step(discriminator, generator, d_optimizer, batch_size, latent_dim, real_images, device):
    """ One training step of the discriminator. """
    
    # Clear the gradients
    d_optimizer.zero_grad()
    
    # Get the discriminator's predictions for the real images
    real_logits = discriminator(real_images)
    
    # Generate fake images
    noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)  # Move noise to the same device as the model
    fake_images = generator(noise).detach()  # Detach the fake images from the generator's graph
    
    # Get the discriminator's predictions for the fake images
    fake_logits = discriminator(fake_images)
    
    # Calculate the discriminator loss
    d_loss = discriminator_loss(real_logits, fake_logits)
    
    # Backpropagate the gradients
    d_loss.backward()
    
    # Update the discriminator's parameters
    d_optimizer.step()
    
    return d_loss.item()


def training(model_G, model_D, z_size, g_optimizer, d_optimizer, generator_loss, discriminator_loss, g_scheduler, d_scheduler, nb_epochs, data_loader, print_every=50, device='cuda'):
    num_epochs = nb_epochs
    samples = []
    losses = []
    sample_size = 16
    fixed_latent_vector = torch.randn(sample_size, z_size, 1, 1).float().to(device)

    model_D.train()
    model_G.train()

    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train the generator 
            g_loss = generator_step(model_G, model_D, g_optimizer, batch_size, z_size, device)

            # Train the discriminator
            d_loss = discriminator_step(model_D, model_G, d_optimizer, batch_size, z_size, real_images, device)

            # Add noise to the input images
            noise_factor = 0.1
            noisy_real_images = real_images + noise_factor * torch.randn_like(real_images).to(device)
            noisy_real_images = torch.clamp(noisy_real_images, 0, 1)

            # Train the discriminator with noisy images
            d_loss_noisy = discriminator_step(model_D, model_G, d_optimizer, batch_size, z_size, noisy_real_images, device)

            # Get the discriminator's predictions for real and fake images
            real_logits = model_D(real_images)
            noise = torch.randn(batch_size, z_size, 1, 1).to(device)
            fake_images = model_G(noise).detach()
            fake_logits = model_D(fake_images)

            # Use label smoothing
            smooth_factor = 0.1
            real_labels = torch.ones_like(real_logits).to(device) - smooth_factor
            fake_labels = torch.zeros_like(fake_logits).to(device) + smooth_factor
            d_loss_smooth = discriminator_loss(real_logits, fake_logits)

            # Backpropagate the gradients for the discriminator with label smoothing
            d_optimizer.zero_grad()
            d_loss_smooth.backward()
            d_optimizer.step()

            if batch_i % print_every == 0:
                # Append discriminator loss and generator loss
                d = d_loss
                g = g_loss
                losses.append((d, g))

                # Print discriminator and generator loss
                time = str(datetime.now()).split('.')[0]
                print(f'{time} | Epoch [{epoch+1}/{num_epochs}] | Batch {batch_i}/{len(data_loader)} | d_loss: {d:.4f} | g_loss: {g:.4f}')

        # Call the schedulers after each epoch
        g_scheduler.step()
        d_scheduler.step()

        # Display images during training
        model_G.eval()
        generated_images = model_G(fixed_latent_vector)
        samples.append(generated_images)
        view_samples(-1, samples)
        model_G.train()

    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses, samples