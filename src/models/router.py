import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class Router(nn.Module):
    """Router network for Mixture of Experts model."""
    
    def __init__(self, input_dim: int, num_experts: int, k: int = 1,
                 capacity_factor: float = 1.0, noise_epsilon: float = 1e-2):
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
            
        return self.router(x)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route input to top-k experts.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of:
            - routing_weights: Tensor of shape [batch_size, num_experts]
            - routing_indices: Tensor of shape [batch_size, k]
            - aux_loss: Load balancing auxiliary loss (None if not training)
        """
        batch_size = x.shape[0]
        routing_scores = self._compute_routing_scores(x)
        
        # Get top-k routing weights and indices
        routing_weights, routing_indices = torch.topk(routing_scores, self.k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            # Compute expert assignment counts
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for idx in routing_indices.view(-1):
                expert_counts[idx] += 1
                
            # Compute load balancing loss
            target_count = torch.ones_like(expert_counts) * (batch_size * self.k / self.num_experts)
            aux_loss = F.mse_loss(expert_counts, target_count)
            
        return routing_weights, routing_indices, aux_loss 