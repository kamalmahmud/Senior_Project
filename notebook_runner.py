import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torchvision.models as models


def clip_cutter(video_path, clip_len, stride):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        all_frames.append(gray)

    cap.release()

    if len(all_frames) < clip_len:
        raise ValueError(
            f"Video has {len(all_frames)} frames, "
            f"which is less than clip_len={clip_len}"
        )

    clips = []
    for start in range(0, total_frames - clip_len + 1, stride):
        clip = all_frames[start: start + clip_len]
        clips.append(np.array(clip))
    return clips


def build_three_channels(clip, smooth_window=3):
    """
    Input:  clip of shape (T, H, W), float32
    Output: three arrays each of shape (T, H, W)
    """
    T = clip
    T_smooth = uniform_filter1d(T, size=smooth_window, axis=0, mode='nearest')
    delta_T = T - T[0]

    return T, T_smooth, delta_T


def stabilize_clip(clip):
    # TODO: ECC alignment — skipped for initial development
    # Each frame aligned to frame 0 using cv2.findTransformECC
    return clip  # passthrough for now


# ── constants ──────────────────────────────────────────────────────────
H, W, T_LEN = 56, 56, 16
PUNCT_X, PUNCT_Y = 28, 38
VEIN_ANGLE = -0.42


# ── clip generator (steps 1–2) ─────────────────────────────────────────
# ── synthetic data generation ──────────────────────────────────────────

def make_background(rng):
    fixed = rng.normal(0, 0.3, size=(H, W)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    skin = (33.5
            + 0.5 * np.sin(xx / 9)
            + 0.3 * np.cos(yy / 8)).astype(np.float32)
    frames = []
    for _ in range(T_LEN):
        temporal = rng.normal(0, 0.1, size=(H, W)).astype(np.float32)
        frames.append(skin + fixed + temporal)
    return frames


def inject_vein_streak(frames):
    yy, xx = np.mgrid[0:H, 0:W]
    ex, ey = xx - PUNCT_X, yy - PUNCT_Y
    dist = np.abs(ex * (-np.sin(VEIN_ANGLE)) + ey * np.cos(VEIN_ANGLE))
    proj = ex * np.cos(VEIN_ANGLE) + ey * np.sin(VEIN_ANGLE)
    mask = (dist < 2.5) & (proj > -2) & (proj < 22)
    cooling = np.where(mask, (1 - dist / 2.5) * 1.5, 0).astype(np.float32)
    return [f.copy() - cooling for f in frames]


def inject_fan(frames, rng):
    fan_dir = rng.uniform(-1.2, -0.2)
    fan_half = rng.uniform(0.5, 0.9)
    max_radius = rng.uniform(12, 20)
    max_cool = rng.uniform(1.5, 3.0)
    onset = rng.randint(2, 6)
    yy, xx = np.mgrid[0:H, 0:W]
    ex, ey = xx - PUNCT_X, yy - PUNCT_Y
    angle = np.arctan2(ey, ex)
    dist = np.sqrt(ex ** 2 + ey ** 2)
    result = []
    for t, f in enumerate(frames):
        frame = f.copy()
        if t >= onset:
            progress = (t - onset) / max(T_LEN - 1 - onset, 1)
            radius_now = max_radius * progress
            angle_diff = np.abs(
                ((angle - fan_dir + np.pi * 3) % (np.pi * 2)) - np.pi
            )
            in_fan = (dist < radius_now) & (dist > 0.5) & (angle_diff < fan_half)
            radial_fade = np.where(in_fan, 1 - dist / np.maximum(radius_now, 1e-6), 0)
            angle_fade = np.where(in_fan, 1 - angle_diff / fan_half, 0)
            frame -= (radial_fade * angle_fade * max_cool).astype(np.float32)
        result.append(frame.astype(np.float32))
    return result


def generate_clip(seed, label):
    rng = np.random.RandomState(seed)
    frames = make_background(rng)
    frames = inject_vein_streak(frames)
    if label == 1:
        frames = inject_fan(frames, rng)
    return np.stack(frames, axis=0)  # (T, H, W)


# ── per-channel resizer ────────────────────────────────────────────────
def resize_channel(channel, size=112):
    return np.stack([
        cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        for frame in channel
    ], axis=0)


# ── dataset ───────────────────────────────────────────────────────────
class ThermalDataset(Dataset):
    def __init__(self, n_samples=1000, pos_ratio=0.2, clip_len=16, img_size=112):
        self.clip_len = clip_len
        self.img_size = img_size
        n_pos = int(n_samples * pos_ratio)
        n_neg = n_samples - n_pos
        self.samples = (
                [(i, 1) for i in range(n_pos)] +
                [(i + n_pos, 0) for i in range(n_neg)]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seed, label = self.samples[idx]
        clip = generate_clip(seed, label)
        T_ch, T_smooth, delta_T = build_three_channels(clip)
        T_ch = resize_channel(T_ch, self.img_size)
        T_smooth = resize_channel(T_smooth, self.img_size)
        delta_T = resize_channel(delta_T, self.img_size)
        tensor = np.stack([T_ch, T_smooth, delta_T], axis=0)
        return (
            torch.tensor(tensor, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


dataset = ThermalDataset(n_samples=100, pos_ratio=0.2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

batch_tensors, batch_labels = next(iter(dataloader))

print("Tensor shape:", batch_tensors.shape)  # expect (4, 3, 16, 112, 112)
print("Labels shape:", batch_labels.shape)  # expect (4,)
print("Label values:", batch_labels)  # mix of 0s and 1s
print("Tensor dtype:", batch_tensors.dtype)  # expect torch.float32
print("Min/max values:", batch_tensors.min().item(),
      batch_tensors.max().item())
print("Any NaN:", torch.isnan(batch_tensors).any().item())  # expect False


class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, fold_div=8):
        super().__init__()
        self.n_segment = n_segment  # T — number of frames
        self.fold_div = fold_div  # 1/8 shifted each direction

    def forward(self, x):
        # x arrives as (B*T, C, H, W) — already reshaped
        BT, C, H, W = x.shape
        B = BT // self.n_segment
        T = self.n_segment
        fold = C // self.fold_div  # number of channels to shift

        x = x.reshape(B, T, C, H, W)
        out = x.clone()

        # Shift fold channels forward: frame t gets features from t-1
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, 0, :fold] = 0  # no previous frame at t=0

        # Shift fold channels backward: frame t gets features from t+1
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        out[:, -1, fold:2 * fold] = 0  # no next frame at last position

        return out.reshape(BT, C, H, W)


class TSMMobileNetV2(nn.Module):
    def __init__(self, n_segment=16, num_classes=2):
        super().__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Insert TSM into each InvertedResidual block
        self.tsm = TemporalShift(n_segment=n_segment)

        # Keep everything except the final classifier
        self.features = backbone.features  # outputs (B*T, 1280, 4, 4)

        # Global average pool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B*T, 1280, 4, 4) → (B*T, 1280, 1, 1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # x: (B, 3, T, 112, 112)
        B, C, T, H, W = x.shape

        # Step 1: reshape for 2D backbone
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)

        # Step 2: apply TSM then backbone
        x = self.tsm(x)  # temporal shift
        x = self.features(x)  # (B*T, 1280, 4, 4)

        # Step 3: spatial pooling
        x = self.pool(x)  # (B*T, 1280, 1, 1)
        x = x.flatten(1)  # (B*T, 1280)

        # Step 4: fold back and average across T
        x = x.reshape(B, T, -1)  # (B, T, 1280)
        x = x.mean(dim=1)  # (B, 1280)

        # Step 5: classify
        x = self.classifier(x)  # (B, 2)
        return x


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
    return auc


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        val_auc = evaluate(model, val_loader, device)
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


model = train(n_epochs=50, patience=10, device=device)
