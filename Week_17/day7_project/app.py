import io
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from PIL import Image
import streamlit as st
from pathlib import Path

from models import Generator, Discriminator, VAE, vae_loss

# ---------- Helpers ----------
@st.cache_resource
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_mnist():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.MNIST("data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    return train, test

def make_grid_img(tensor_bchw, nrow=8, normalize=True):
    grid = utils.make_grid(tensor_bchw, nrow=nrow, normalize=normalize, value_range=(0,1))
    img = (grid.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    return Image.fromarray(img)

def laplacian_variance(img_batch):
    # crude sharpness proxy; expects [B,1,28,28] scaled 0..1
    k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=img_batch.device).view(1,1,3,3)
    edges = F.conv2d(img_batch, k, padding=1)
    return edges.var(dim=[1,2,3]).mean().item()

def pixel_variance(img_batch):
    return img_batch.var(dim=[0,2,3]).mean().item()

def bytes_for_download(pil_img, filename="samples.png"):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def load_weights_if_exist(model, path):
    if Path(path).exists():
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        return ckpt
    return None

# ---------- UI ----------
st.set_page_config(page_title="AI Art Generator — GAN & VAE", layout="wide")
st.title("🧪 AI Art Generator (MNIST) — GAN & VAE")

device = get_device()
train_ds, test_ds = load_mnist()
colL, colR = st.columns([2,1])

with colR:
    st.subheader("Controls")
    model_choice = st.selectbox("Model", ["GAN", "VAE"])
    if model_choice == "GAN":
        latent_dim = st.slider("Latent dim (z)", 8, 128, 32, step=8)
    else:
        latent_dim = st.slider("Latent dim (z)", 2, 32, 10, step=2)
    n_samples = st.slider("Samples", 8, 64, 32, step=8)
    do_interpolate = st.checkbox("Interpolate (z₁→z₂)", value=False)
    do_reconstruct = st.checkbox("Reconstruct real digits (VAE only)", value=True if model_choice=="VAE" else False, disabled=(model_choice=="GAN"))
    steps = st.slider("Interpolation steps", 2, 16, 8, disabled=not do_interpolate)

    st.markdown("---")
    st.subheader("Training")
    mini_train = st.button("⚡ Mini-Train (quick 3 epochs)")
    full_train = st.button("🧠 Train 10 epochs")

with colL:
    st.write("Device:", device)

# ---------- Models ----------
if model_choice == "GAN":
    G = Generator(latent_dim=latent_dim).to(device)
    ckpt = load_weights_if_exist(G, "weights/gan.pt")
    if ckpt and ckpt.get("latent_dim", latent_dim) != latent_dim:
        st.info(f"Loaded weights have latent_dim={ckpt['latent_dim']}. New model uses {latent_dim}. You can still generate (random init) or retrain.")
    model = G
else:
    vae = VAE(latent_dim=latent_dim).to(device)
    ckpt = load_weights_if_exist(vae, "weights/vae.pt")
    if ckpt and ckpt.get("latent_dim", latent_dim) != latent_dim:
        st.info(f"Loaded weights have latent_dim={ckpt['latent_dim']}. New model uses {latent_dim}. You can still generate (random init) or retrain.")
    model = vae

# ---------- Optional training from UI ----------
# ---------- Optional training from UI (FIXED) ----------
def train_vae_epochs(epochs=3, lr=1e-3, batch_size=128):
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    progress = st.progress(0)
    status = st.empty()
    with st.spinner("Training VAE..."):
        for ep in range(epochs):
            vae.train()
            for x, _ in loader:
                x = x.to(device)
                recon, mu, logvar = vae(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
            status.write(f"VAE epoch {ep+1}/{epochs} - loss {loss.item():.2f}")
            progress.progress((ep + 1) / epochs)

    Path("weights").mkdir(exist_ok=True)
    torch.save({"state_dict": vae.state_dict(), "latent_dim": vae.latent_dim}, "weights/vae.pt")
    status.write("✅ VAE training complete. Saved to weights/vae.pt")


def train_gan_epochs(epochs=3, lr=2e-4, batch_size=128):
    # fresh D for training; keep G from outer scope
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = torch.nn.BCELoss()
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    progress = st.progress(0)
    status = st.empty()
    with st.spinner("Training GAN..."):
        for ep in range(epochs):
            for real, _ in loader:
                real = real.to(device)
                bsz = real.size(0)
                real_lbl = torch.ones(bsz, device=device)
                fake_lbl = torch.zeros(bsz, device=device)

                # sample noise + generate
                z = torch.randn(bsz, G.latent_dim, device=device)
                fake = G(z)

                # ---- Train D
                D_real = D(real); D_fake = D(fake.detach())
                loss_D = bce(D_real, real_lbl) + bce(D_fake, fake_lbl)
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

                # ---- Train G
                D_fake = D(fake)
                loss_G = bce(D_fake, real_lbl)
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            status.write(f"GAN epoch {ep+1}/{epochs} — D: {loss_D.item():.3f} | G: {loss_G.item():.3f}")
            progress.progress((ep + 1) / epochs)

    Path("weights").mkdir(exist_ok=True)
    torch.save({"state_dict": G.state_dict(), "latent_dim": G.latent_dim}, "weights/gan.pt")
    status.write("✅ GAN training complete. Saved to weights/gan.pt")

if mini_train:
    if model_choice == "VAE": train_vae_epochs(epochs=3)
    else: train_gan_epochs(epochs=3)

if full_train:
    if model_choice == "VAE": train_vae_epochs(epochs=10)
    else: train_gan_epochs(epochs=10)

# ---------- Generation ----------
st.subheader("Results")

if model_choice == "GAN":
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        samples = (model(z).clamp(-1,1) + 1)/2
    pil = make_grid_img(samples, nrow=8, normalize=False)
    st.image(pil, caption="GAN Samples", use_column_width=True)

    if do_interpolate:
        z1, z2 = torch.randn(1, model.latent_dim, device=device), torch.randn(1, model.latent_dim, device=device)
        zs = [ (1-a)*z1 + a*z2 for a in torch.linspace(0,1,steps, device=device) ]
        with torch.no_grad():
            imgs = torch.cat([ (model(z).clamp(-1,1)+1)/2 for z in zs ], dim=0)
        pil2 = make_grid_img(imgs, nrow=steps, normalize=False)
        st.image(pil2, caption="GAN Latent Interpolation", use_column_width=True)

    # Metrics
    st.markdown("**Metrics (proxies):**")
    s_div = pixel_variance(samples)
    s_sharp = laplacian_variance(samples)
    st.write(f"Diversity (pixel variance): **{s_div:.4f}**, Sharpness (Laplacian var): **{s_sharp:.4f}**")

    st.download_button("Download samples.png", data=bytes_for_download(pil), file_name="gan_samples.png", mime="image/png")

else:  # VAE
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        samples = (model.decode(z).clamp(-1,1) + 1)/2
    pil = make_grid_img(samples, nrow=8, normalize=False)
    st.image(pil, caption="VAE Samples", use_column_width=True)

    if do_reconstruct:
        # Show reconstructions for 8 test digits
        loader_test = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=True)
        x, y = next(iter(loader_test))
        x = x.to(device)
        with torch.no_grad():
            recon, _, _ = model(x)
            vis = torch.cat([ (x+1)/2, (recon.clamp(-1,1)+1)/2 ], dim=0)
        pil_rec = make_grid_img(vis, nrow=8, normalize=False)
        st.image(pil_rec, caption="VAE Reconstructions (top: input, bottom: recon)", use_column_width=True)

    if do_interpolate:
        # Interpolate in latent space between two encoded digits
        loader_test = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=True)
        x2, _ = next(iter(loader_test))
        x2 = x2.to(device)
        with torch.no_grad():
            mu, logvar = model.encode(x2)
            z1 = mu[0:1]
            z2 = mu[1:2]
            zs = [ (1-a)*z1 + a*z2 for a in torch.linspace(0,1,steps, device=device) ]
            imgs = torch.cat([ (model.decode(z).clamp(-1,1)+1)/2 for z in zs ], dim=0)
        pil_interp = make_grid_img(imgs, nrow=steps, normalize=False)
        st.image(pil_interp, caption="VAE Latent Interpolation", use_column_width=True)

    # Metrics
    st.markdown("**Metrics (proxies):**")
    s_div = pixel_variance(samples)
    s_sharp = laplacian_variance(samples)
    st.write(f"Diversity (pixel variance): **{s_div:.4f}**, Sharpness (Laplacian var): **{s_sharp:.4f}**")

    st.download_button("Download samples.png", data=bytes_for_download(pil), file_name="vae_samples.png", mime="image/png")

st.caption("Tip: For better quality, train 10–20 epochs, then reload the app.")
