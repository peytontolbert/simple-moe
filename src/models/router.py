import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from . import _model_stack


class Router(nn.Module):
    """Router network for Mixture of Experts model."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int = 1,
                 capacity_factor: float = 1.0, noise_epsilon: float = 1e-2,
                 drop_policy: str = "dropless"):
        """
        Initialize the router network.
        
        Args:
            input_dim: Dimension of input features
            num_experts: Number of experts
            k: Number of experts to route to
            capacity_factor: Multiplier for expert capacity
            noise_epsilon: Noise factor for load balancing
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.noise_epsilon = noise_epsilon
        self.drop_policy = drop_policy
        
        # Router weights
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Expert capacity: batch_size * capacity_factor * (k / num_experts)
        self.capacity = lambda batch_size: int(batch_size * capacity_factor * k / num_experts)
        
    def _compute_routing_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities for each expert."""
        # Add noise during training for better load balancing
        if self.training:
            noise = torch.randn_like(x) * self.noise_epsilon
            x = x + noise
            
        return _model_stack.runtime_linear(self.router, x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route input to top-k experts.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of:
            - routing_weights: Tensor of shape [batch_size, k]
            - routing_indices: Tensor of shape [batch_size, k]
            - aux_loss: Load balancing auxiliary loss (None if not training)
        """
        batch_size = x.shape[0]
        routing_scores = self._compute_routing_scores(x)
        
        # Get top-k routing weights and indices
        routing_weights, routing_indices = _model_stack.topk_route(
            routing_scores,
            k=self.k,
            capacity_factor=self.capacity_factor,
            drop_policy=self.drop_policy,
        )
        
        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            # Compute expert assignment counts
            expert_counts = torch.zeros(
                self.num_experts,
                device=x.device,
                dtype=routing_scores.dtype,
            )
            expert_counts.scatter_add_(
                0,
                routing_indices.reshape(-1),
                torch.ones(
                    routing_indices.numel(),
                    device=x.device,
                    dtype=routing_scores.dtype,
                ),
            )
                
            # Compute load balancing loss
            target_count = torch.ones_like(expert_counts) * (batch_size * self.k / self.num_experts)
            aux_loss = F.mse_loss(expert_counts, target_count)
            
        return routing_weights, routing_indices, aux_loss
