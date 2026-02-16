import torch
from ivthermal.config_schema import load_config
from ivthermal.data.dummy import make_dummy_dataloader_from_app_config
from ivthermal.contracts import assert_input_tensor

def test_dummy_dataloader_batch_shapes():
    cfg = load_config("config.yaml")
    cfg.train.batch_size = 4  # small for test

    loader = make_dummy_dataloader_from_app_config(cfg, split="train", num_samples=16)
    x, y = next(iter(loader))

    # x contract
    assert_input_tensor(x, expected_T=cfg.clip.T)
    assert x.dtype == torch.float32

    # y shape/dtype
    if y.ndim == 1:
        y = y.view(-1, 1)
    assert y.shape == (x.shape[0], 1)
    assert y.dtype == torch.float32
    # values in {0,1}
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})
