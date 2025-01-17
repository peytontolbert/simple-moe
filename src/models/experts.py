import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class ExpertBase(nn.Module):
    """Base class for expert networks in the Mixture of Experts model."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize the expert network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dim: Dimension of hidden layer (if None, uses 4x input_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or 4 * input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get expert configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim
        }

class FFNExpert(ExpertBase):
    """Feed-forward neural network expert implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the FFN expert.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__(input_dim, output_dim, hidden_dim)
        
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = self.hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'num_layers': len(self.network) // 3 + 1,
            'dropout': self.network[2].p if len(self.network) > 2 else 0.0
        })
        return config 