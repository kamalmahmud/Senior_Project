import torch
from ivthermal.contracts import assert_input_tensor, assert_output_logits, ContractViolation

def test_input_ok():
    x = torch.randn(2, 3, 16, 112, 112, dtype=torch.float32)
    assert_input_tensor(x, expected_T=16)

def test_input_bad_shape():
    x = torch.randn(2, 3, 16, 110, 112, dtype=torch.float32)
    try:
        assert_input_tensor(x, expected_T=16)
        assert False, "Expected ContractViolation"
    except ContractViolation:
        pass

def test_output_ok():
    y = torch.randn(2, 1, dtype=torch.float32)
    assert_output_logits(y, batch_size=2)

def test_output_bad_shape():
    y = torch.randn(2, 2, dtype=torch.float32)
    try:
        assert_output_logits(y, batch_size=2)
        assert False, "Expected ContractViolation"
    except ContractViolation:
        pass
