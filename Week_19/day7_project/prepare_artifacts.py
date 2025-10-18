# prepare_artifacts.py
# Trains a small Teacher (baseline), quantizes it, and distills a Student.
# CPU-only, fast runs to produce demo artifacts for the Streamlit app.

import os, time, copy, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

torch.set_num_threads(4)
device = torch.device("cpu")

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

def get_data(subset=10000, batch_train=256, batch_test=512):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    test = datasets.MNIST(root="./data", train=False, transform=tfm, download=True)
    if subset and subset < len(train_full):
        idx = torch.randperm(len(train_full))[:subset].tolist()
        train = torch.utils.data.Subset(train_full, idx)
    else:
        train = train_full
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_train, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_test, shuffle=False, num_workers=2)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
    return correct / max(n, 1)

def train_epoch(model, loader, opt, loss_fn, cap_batches=200):
    model.train()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i == cap_batches: break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        total += loss.item() * y.size(0); n += y.size(0)
    return total / max(n, 1)

def distill_epoch(student, teacher, loader, opt, ce, alpha=0.3, T=2.0, cap_batches=200):
    student.train(); teacher.eval()
    kl = nn.KLDivLoss(reduction="batchmean")
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i == cap_batches: break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_soft = torch.softmax(teacher(x)/T, dim=1)
        s_logits = student(x)/T
        loss_soft = kl(torch.log_softmax(s_logits, dim=1), t_soft) * (T*T)
        loss = (1 - alpha) * loss_soft + alpha * ce(student(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * y.size(0); n += y.size(0)
    return total / max(n, 1)

def main(epochs_teacher=2, epochs_student=2, subset=10000):
    train_loader, test_loader = get_data(subset=subset)
    ce = nn.CrossEntropyLoss()

    # Train Teacher
    teacher = TeacherNet().to(device)
    opt_t = optim.Adam(teacher.parameters(), lr=1e-3)
    for ep in range(epochs_teacher):
        loss = train_epoch(teacher, train_loader, opt_t, ce)
        acc = evaluate(teacher, test_loader)
        print(f"[Teacher] Epoch {ep+1}: loss={loss:.4f} acc={acc:.3f}")
    torch.save(teacher.state_dict(), ART/"teacher_baseline.pth")

    # Quantize Teacher
    qteacher = torch.quantization.quantize_dynamic(copy.deepcopy(teacher), {nn.Linear}, dtype=torch.qint8)
    torch.save(qteacher.state_dict(), ART/"teacher_quantized.pth")

    # Distill Student
    student = StudentNet().to(device)
    opt_s = optim.Adam(student.parameters(), lr=1e-3)
    for ep in range(epochs_student):
        dloss = distill_epoch(student, teacher, train_loader, opt_s, ce, alpha=0.3, T=2.0)
        acc_s = evaluate(student, test_loader)
        print(f"[Student] Epoch {ep+1}: distill_loss={dloss:.4f} acc={acc_s:.3f}")
    torch.save(student.state_dict(), ART/"student_distilled.pth")

    print("Artifacts written to:", ART.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_teacher", type=int, default=2)
    parser.add_argument("--epochs_student", type=int, default=2)
    parser.add_argument("--subset", type=int, default=10000)
    args = parser.parse_args()
    main(args.epochs_teacher, args.epochs_student, args.subset)
