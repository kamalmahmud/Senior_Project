import torch
import torch.nn as nn
from data.dataset import ThermalDataset
from models.mobilenet import TSMMobileNetV2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from metrics import find_best_threshold, compute_metrics, print_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device)
            outputs = model(tensors)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    best_thr = find_best_threshold(all_labels, all_probs, target_fa_h=0.3)
    metrics = compute_metrics(all_labels, all_probs, threshold=best_thr)
    return auc


def train(n_epochs=50, patience=10, device='cpu'):
    # Datasets
    train_dataset = ThermalDataset(n_samples=800, pos_ratio=0.2)
    val_dataset = ThermalDataset(n_samples=200, pos_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8,
                            shuffle=False, num_workers=0)

    # Model
    class_weights = torch.tensor([1.0, 4.0]).to(device)
    model = TSMMobileNetV2(n_segment=16, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    best_auc = 0.0
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device)

        metrics = evaluate(model, val_loader, device)
        val_auc = metrics["auc"]
        scheduler.step(val_auc)
        print_metrics(metrics)
        scheduler.step(val_auc)

        print(f"Epoch {epoch + 1:3d} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}. "
                  f"Best AUC: {best_auc:.4f}")
            break

            # Restore best weights
    model.load_state_dict(best_weights)
    return model
