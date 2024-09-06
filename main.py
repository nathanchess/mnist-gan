import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.ReLU_slope = 0.2

        # Usage of Leaky ReLU is to prevent dead neurons, which can be caused by ReLU since it outputs 0 for all negatives. Leaky ReLU outputs a small value for negative inputs, which prevents this and allows the neuron to learn.
        # Usage of nn.Sigmoid() is to transform the last node into a probability, which is necessary for binary classification.
        self.layers = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(self.ReLU_slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(self.ReLU_slope),
            nn.Linear(64, 32),
            nn.LeakyReLU(self.ReLU_slope),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super().__init__()

        # Usage of ReLU instead of Leaky ReLU is to allow the generator to come up with more diverse patterns. This allows the gradients to flow easily, making learning efficient.
        # Usage of nn.Tanh
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
    
class NumberGAN():

    def __init__(self, learning_rate=0.0001, z_dim=64, image_dim=28*28*1, batch_size=32, epochs=50):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(self.device)

        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.discriminator = Discriminator(image_dim).to(self.device)
        self.generator = Generator(z_dim, image_dim).to(self.device)

        self.fixed_noise = torch.randn((batch_size, z_dim)).to(self.device)

        # transforms.Compose performs a linear set of transformations on vectors.
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5   ,))]
        )

        self.dataset = datasets.MNIST(root='dataset/', transform=self.transforms, download=True)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # optim.Adam is used due to it's adaptive learning rate, which allows the model to converge faster.
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=learning_rate)
        
        # BCELoss stands for binary cross entropy loss, which is typically used for binary classification problems. In our case  it is if the discriminator can catch the fake made by the generator model.
        self.criterion = nn.BCELoss()

        self.write_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
        self.write_real = SummaryWriter(f'runs/GAN_MNIST/real')

    def train(self):

        step = 0

        for epoch in range(self.epochs):
            for i, (real, _) in enumerate(self.loader):
                real = real.view(-1, self.image_dim).to(self.device)
                batch_size = real.shape[0]
                
                # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = torch.randn(self.batch_size, self.z_dim).to(self.device) # z
                fake = self.generator(noise) # G(z)
                disc_real = self.discriminator(real).view(-1).to(self.device) # D(x)

                 # disc_real outputs a tensor for the probability it is a real image and we compare it with torch.ones_like since 1 represents a real image and we are passing in real images.
                lossD_real = self.criterion(disc_real, torch.ones_like(disc_real))

                disc_fake = self.discriminator(fake).view(-1).to(self.device) # D(G(z))
                lossD_fake = self.criterion(disc_fake, torch.zeros_like(disc_fake))

                # Average the losses as it is important for discriminator to understand the error in both identifying fake and real images.
                lossD = (lossD_real + lossD_fake) / 2

                # Reset all gradients to zero within discriminator so they don't affect previous calculations.
                self.discriminator.zero_grad()

                # Computes gradients (slope) based on averaged loss for both real and fake images and stores them in .grad attribute
                lossD.backward(retain_graph=True)

                # Updates the parameters based on the gradients computed by .backward()
                self.optimizer_discriminator.step()

                ## Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                output = self.discriminator(fake).view(-1).to(self.device)
                lossG = self.criterion(output, torch.ones_like(disc_fake)) # Maximize D(G(z)) by using comparison with 1.
                self.generator.zero_grad()
                lossG.backward()
                self.optimizer_generator.step()

                if i == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}] Batch {i}/{len(self.loader)} \
                            Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                    )

                    with torch.no_grad():
                        fake = self.generator(noise).reshape(-1, 1, 28, 28)
                        data = real.reshape(-1, 1, 28, 28)
                        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                        img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                        self.write_fake.add_image(
                            "Mnist Fake Images", img_grid_fake, global_step=step
                        )
                        self.write_real.add_image(
                            "Mnist Real Images", img_grid_real, global_step=step
                        )
                        step += 1

newGan = NumberGAN()
newGan.train()