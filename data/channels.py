from scipy.ndimage import uniform_filter1d
import cv2
import numpy as np


def build_three_channels(clip, smooth_window=3):
    """
    Input:  clip of shape (T, H, W), float32
    Output: three arrays each of shape (T, H, W)
    """
    T = clip
    T_smooth = uniform_filter1d(T, size=smooth_window, axis=0, mode='nearest')
    delta_T = T - T[0]

    return T, T_smooth, delta_T


def resize_channel(channel, size=112):
    return np.stack([
        cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        for frame in channel
    ], axis=0)
