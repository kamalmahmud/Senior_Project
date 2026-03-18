import torch.nn as nn


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
