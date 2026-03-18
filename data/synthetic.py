import numpy as np
from config import H, W, T_LEN, PUNCT_Y, PUNCT_X, VEIN_ANGLE


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
