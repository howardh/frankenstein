import torch

from frankenstein.math.prefix_sum import prefix_sum__reference
from frankenstein.math.prefix_sum import prefix_sum__parallel


def test_prefix_sum():
    data = torch.tensor([1, 2, 3, 4, 5])
    output = prefix_sum__parallel(data)
    assert torch.allclose(output, torch.tensor([1, 3, 6, 10, 15]))

def test_prefix_sum_random():
    data = torch.randn(100)
    output_parallel = prefix_sum__parallel(data)
    output_reference = prefix_sum__reference(data)
    assert torch.allclose(output_parallel, output_reference)
