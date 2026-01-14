from flax import linen as nn
from jax import numpy as jnp
from typing import Dict, Any, Optional

class ExpertBase(nn.Module):
    """Base class for expert networks in the Mixture of Experts model."""
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the expert."""
        hidden_dim = self.hidden_dim or 4 * self.input_dim
        x = nn.Dense(features=hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get expert configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim
        }

class FFNExpert(nn.Module):
    """Feed-forward neural network expert implementation."""
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    num_layers: int = 2
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """Forward pass through the FFN expert."""
        hidden_dim = self.hidden_dim or 4 * self.input_dim
        
        for i in range(self.num_layers - 1):
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not training, rng=self.make_rng('dropout'))
            
        x = nn.Dense(features=self.output_dim)(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Get expert configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }