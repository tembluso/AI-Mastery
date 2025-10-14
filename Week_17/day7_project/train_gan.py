import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path
from models import Generator, Discriminator

def main(epochs=10, latent_dim=32, batch_size=128, lr=2e-4, betas=(0.5,0.999), seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST("data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    G, D = Generator(latent_dim).to(device), Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCELoss()

    for epoch in range(epochs):
        for real, _ in loader:
            real = real.to(device)
            bsz = real.size(0)
            real_lbl = torch.ones(bsz, device=device)
            fake_lbl = torch.zeros(bsz, device=device)

            # --- Train D ---
            z = torch.randn(bsz, latent_dim, device=device)
            fake = G(z)
            D_real = D(real)
            D_fake = D(fake.detach())
            loss_D = bce(D_real, real_lbl) + bce(D_fake, fake_lbl)
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # --- Train G ---
            D_fake = D(fake)
            loss_G = bce(D_fake, real_lbl)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        print(f"[GAN] Epoch {epoch+1}/{epochs} - D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")

    Path("weights").mkdir(exist_ok=True)
    torch.save({"state_dict": G.state_dict(), "latent_dim": latent_dim}, "weights/gan.pt")
    print("Saved weights/gan.pt")

if __name__ == "__main__":
    main()
