from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl

# helper function for viewing a list of passed-in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(14, 4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu()  # Move tensor to CPU
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)).cpu().numpy(), cmap='Greys_r')  # Convert to NumPy array
    plt.show()

def training(model_G, model_D, z_size, g_optimizer, d_optimizer, real_loss, fake_loss, g_scheduler, d_scheduler, nb_epochs, data_loader, device='cuda'):
    # training hyperparams
    num_epochs = nb_epochs

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 100

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = torch.randn((sample_size, z_size)).to(device)

    # train the network
    model_D.train()
    model_G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)

            batch_size = real_images.size(0)

            ## Important rescaling step ##
            real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================

            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            # smooth the real labels
            D_real = model_D(real_images)
            d_real_loss = real_loss(D_real, smooth=True)

            # 2. Train with fake images

            # Generate fake images
            # gradients don't have to flow during this step
            with torch.no_grad():
                z = torch.randn((batch_size, z_size)).to(device)
                fake_images = model_G(z)

            # Compute the discriminator losses on fake images
            D_fake = model_D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = torch.randn((batch_size, z_size)).to(device)
            fake_images = model_G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = model_D(fake_images)
            g_loss = real_loss(D_fake)  # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                time = str(datetime.now()).split('.')[0]
                print(f'{time} | Epoch [{epoch+1}/{num_epochs}] | Batch {batch_i}/{len(data_loader)} | d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}')
                
                ## AFTER EACH EPOCH##
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))

        # Call the scheduler after the optimizer.step()
        g_scheduler.step()
        d_scheduler.step()

        # generate and save sample, fake images
        model_G.eval()  # eval mode for generating samples
        samples_z = model_G(fixed_z)
        samples.append(samples_z)
        # Assuming view_samples is a function to visualize generated samples
        view_samples(-1, samples)
        model_G.train()  # back to train mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses, samples

