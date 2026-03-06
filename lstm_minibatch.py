import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import imdb
from keras.preprocessing import sequence

TOP_WORDS = 5000
MAX_LEN   = 500
EMBED_DIM = 32
HIDDEN    = 100
EPOCHS    = 40
BATCH     = 64
SEED      = 42
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dữ liệu imdb
(X_tr, y_tr), (X_te, y_te) = imdb.load_data(num_words=TOP_WORDS)
X_tr = sequence.pad_sequences(X_tr, maxlen=MAX_LEN)
X_te = sequence.pad_sequences(X_te, maxlen=MAX_LEN)

X_small = torch.tensor(X_tr[:2000], dtype=torch.long)
y_small = torch.tensor(y_tr[:2000], dtype=torch.float32)
X_val   = torch.tensor(X_te[:1000], dtype=torch.long)
y_val   = torch.tensor(y_te[:1000], dtype=torch.float32)

# shuffle=False + generator cố định => cả hai method sẽ sử dụng batch giống nhau
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(
    TensorDataset(X_small, y_small),
    batch_size=BATCH,
    shuffle=True,
    generator=g
)


# ── Model ─────────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(TOP_WORDS, EMBED_DIM, padding_idx=0)
        self.lstm  = nn.LSTM(EMBED_DIM, HIDDEN, batch_first=True)
        self.fc    = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h.squeeze(0))).squeeze(1)


def get_accuracy(model, X, y, batch_size=128):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            preds = model(xb) > 0.5
            correct += (preds.cpu().float() == y[i:i+batch_size]).float().sum().item()
    return correct / len(y)


def get_loss(model, X, y, batch_size=128):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            yb = y[i:i+batch_size].to(DEVICE)
            total_loss += nn.BCELoss()(model(xb), yb).item() * len(xb)
    return total_loss / len(y)


def batched_loss(model, X, y, batch_size=128):
    """Tính loss trên toàn bộ dataset theo batch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            yb = y[i:i+batch_size].to(DEVICE)
            total_loss += nn.BCELoss()(model(xb), yb).item() * len(xb)
    return total_loss / len(y)


# Khởi tạo một lần, cả hai method dùng chung init_state
torch.manual_seed(SEED)
_init_model = SentimentLSTM().to(DEVICE)
init_state  = {k: v.clone() for k, v in _init_model.state_dict().items()}


# ── Adam ──────────────────────────────────────────────────
print("=" * 60)
print("ADAM OPTIMIZER")
print("=" * 60)

model_adam = SentimentLSTM().to(DEVICE)
model_adam.load_state_dict(init_state)
optimizer = torch.optim.Adam(model_adam.parameters())
criterion = nn.BCELoss()

t0 = time.time()
best_acc_adam   = 0.0
best_loss_adam  = float('inf')
best_epoch_adam = 0

# Reset generator để Adam thấy batch shuffle theo đúng thứ tự cố định
g.manual_seed(SEED)
for epoch in range(EPOCHS):
    model_adam.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_adam(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    val_acc  = get_accuracy(model_adam, X_val, y_val)
    val_loss = get_loss(model_adam, X_val, y_val)
    if val_acc > best_acc_adam:
        best_acc_adam   = val_acc
        best_loss_adam  = val_loss
        best_epoch_adam = epoch + 1
    print(f"  Epoch {epoch+1:>2}: train_loss={running/len(train_loader):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

t_adam = time.time() - t0
print(f"\nAdam -> best epoch {best_epoch_adam}: loss={best_loss_adam:.4f} | acc={best_acc_adam*100:.2f}% | time={t_adam:.1f}s")

# Giải phóng GPU
del model_adam, optimizer, criterion
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── QuickProp ─────────────────────────────────────────────
#
# Công thức QuickProp (Fahlman, 1989):
#   Δw(t) = g(t) / (g(t-1) - g(t)) * Δw(t-1)
#
# Bước đầu tiên dùng GD để khởi tạo prev_grad và prev_delta.

print("\n" + "=" * 60)
print("QUICKPROP OPTIMIZER")
print("=" * 60)


class QuickPropOptimizer:
    def __init__(self, parameters, lr=0.01, max_delta=5.0):
        self.lr = lr
        self.max_delta = max_delta
        self.params = list(parameters)
        self.prev_grad  = [None] * len(self.params)
        self.prev_delta = [None] * len(self.params)
        self.step_count = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data.clone()

            if self.prev_grad[i] is None:
                # Bước 1: bootstrap bằng GD
                delta = -self.lr * grad
            else:
                denom = self.prev_grad[i] - grad
                # QuickProp khi mẫu số đủ lớn
                qp_mask = denom.abs() > 1e-7
                delta = torch.zeros_like(grad)
                delta[qp_mask] = grad[qp_mask] / denom[qp_mask] * self.prev_delta[i][qp_mask]
                # Fallback GD khi mẫu số quá nhỏ
                delta[~qp_mask] = -self.lr * grad[~qp_mask]

            # Clamp để tránh bùng nổ
            delta = delta.clamp(-self.max_delta, self.max_delta)

            self.prev_grad[i]  = grad
            self.prev_delta[i] = delta.clone()
            p.data.add_(delta)


model_qp = SentimentLSTM().to(DEVICE)
model_qp.load_state_dict(init_state)   # cùng weight khởi tạo với Adam
criterion_qp = nn.BCELoss()
qp_optim = QuickPropOptimizer(model_qp.parameters(), lr=0.01, max_delta=5.0)

print(f"\nTrain from scratch, same init weights as Adam:")
t1 = time.time()
best_acc_qp   = 0.0
best_loss_qp  = float('inf')
best_epoch_qp = 0

g.manual_seed(SEED)
for epoch in range(EPOCHS):
    model_qp.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        qp_optim.zero_grad()
        loss = criterion_qp(model_qp(xb), yb)
        loss.backward()
        qp_optim.step()
        running += loss.item()
    val_acc  = get_accuracy(model_qp, X_val, y_val)
    val_loss = get_loss(model_qp, X_val, y_val)
    if val_acc > best_acc_qp:
        best_acc_qp   = val_acc
        best_loss_qp  = val_loss
        best_epoch_qp = epoch + 1
    print(f"  Epoch {epoch+1:>2}: train_loss={running/len(train_loader):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

t_qp = time.time() - t1
print(f"\nQuickProp -> best epoch {best_epoch_qp}: loss={best_loss_qp:.4f} | acc={best_acc_qp*100:.2f}% | time={t_qp:.1f}s")


# ── So sánh ───────────────────────────────────────────────
print("\n" + "=" * 62)
print(f"{'Method':<24} {'Best Loss':>10} {'Best Acc':>10} {'Epoch':>6} {'Time':>8}")
print("-" * 62)
print(f"{'Adam':<24} {best_loss_adam:>10.4f} {best_acc_adam*100:>9.2f}% {best_epoch_adam:>6} {t_adam:>7.1f}s")
print(f"{'QuickProp':<24} {best_loss_qp:>10.4f} {best_acc_qp*100:>9.2f}% {best_epoch_qp:>6} {t_qp:>7.1f}s")
print("=" * 62)