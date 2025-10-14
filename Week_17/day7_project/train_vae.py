import torch, torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path
from models import VAE, vae_loss

def main(epochs=10, latent_dim=10, batch_size=128, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST("data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    vae = VAE(latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        vae.train()
        for x, _ in loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[VAE] Epoch {epoch+1}/{epochs} - loss: {loss.item():.2f}")

    Path("weights").mkdir(exist_ok=True)
    torch.save({"state_dict": vae.state_dict(), "latent_dim": latent_dim}, "weights/vae.pt")
    print("Saved weights/vae.pt")

if __name__ == "__main__":
    main()
