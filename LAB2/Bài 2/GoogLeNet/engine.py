import torch
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, running = 0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * yb.size(0)
        total += yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()

    return running / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running += loss.item() * yb.size(0)
        total += yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()

        y_true += yb.cpu().tolist()
        y_pred += logits.argmax(1).cpu().tolist()

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return running / total, correct / total, macro_f1, y_true, y_pred