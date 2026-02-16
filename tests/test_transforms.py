import torch
from ivthermal.data.transforms import (
    ThermalThreeChannelBuilder,
    temporal_smooth_cthw,
    deltaT_cthw,
    TransformError,
)

def test_deltaT_first_frame():
    # CTHW with C=1
    t_raw = torch.tensor([[
        [[1.0, 2.0],
         [3.0, 4.0]],
        [[2.0, 3.0],
         [4.0, 5.0]],
        [[4.0, 6.0],
         [8.0, 10.0]],
    ]])  # shape (1,T=3,H=2,W=2)

    dT = deltaT_cthw(t_raw, baseline="first_frame")
    # first frame becomes zeros
    assert torch.allclose(dT[:, 0], torch.zeros_like(dT[:, 0]))
    # second frame is raw - first
    assert torch.allclose(dT[:, 1], t_raw[:, 1] - t_raw[:, 0])

def test_temporal_smooth_window1_identity():
    t_raw = torch.randn(1, 8, 4, 4)
    y = temporal_smooth_cthw(t_raw, window=1)
    assert torch.allclose(y, t_raw)

def test_temporal_smooth_replicate_padding_behavior():
    # Simple sequence where replicate padding matters.
    # T=3, values per time: [0, 10, 20]
    t_raw = torch.zeros(1, 3, 1, 1)
    t_raw[:, 0, 0, 0] = 0.0
    t_raw[:, 1, 0, 0] = 10.0
    t_raw[:, 2, 0, 0] = 20.0

    # window=3 -> pad=1 replicate:
    # t=0 uses [0,0,10] -> 10/3
    # t=1 uses [0,10,20] -> 30/3=10
    # t=2 uses [10,20,20] -> 50/3
    y = temporal_smooth_cthw(t_raw, window=3)
    assert torch.isclose(y[:, 0, 0, 0], torch.tensor(10.0/3.0))
    assert torch.isclose(y[:, 1, 0, 0], torch.tensor(10.0))
    assert torch.isclose(y[:, 2, 0, 0], torch.tensor(50.0/3.0))

def test_builder_outputs_three_channels_cthw():
    builder = ThermalThreeChannelBuilder(smooth_window=5, deltaT_baseline="first_frame")
    t_raw = torch.randn(1, 16, 112, 112)
    out = builder(t_raw)
    assert out.shape == (3, 16, 112, 112)  # (3,T,H,W)

def test_builder_outputs_three_channels_ncthw():
    builder = ThermalThreeChannelBuilder(smooth_window=5, deltaT_baseline="first_frame")
    t_raw = torch.randn(2, 1, 16, 112, 112)
    out = builder(t_raw)
    assert out.shape == (2, 3, 16, 112, 112)  # (B,3,T,H,W)

def test_builder_rejects_wrong_channel_count():
    builder = ThermalThreeChannelBuilder()
    t_raw = torch.randn(2, 2, 16, 112, 112)  # C=2 invalid
    try:
        builder(t_raw)
        assert False, "Expected TransformError"
    except TransformError:
        pass
