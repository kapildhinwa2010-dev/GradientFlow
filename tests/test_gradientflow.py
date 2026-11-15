import pytest
import torch
from gradientflow import GradientFlow, create_gradientflow

class TestGradientFlow:
    def test_init(self):
        m = GradientFlow(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = GradientFlow(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_gradientflow()
        assert isinstance(m, GradientFlow)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
