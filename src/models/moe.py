import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple

from .experts import ExpertBase, FFNExpert
from .router import Router

class MixtureOfExperts(nn.Module):
    """Mixture of Experts model implementation."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_experts: int,
                 expert_class: type = FFNExpert,
                 expert_kwargs: Optional[Dict[str, Any]] = None,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 router_noise_epsilon: float = 1e-2):
        """
        Initialize the MoE model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            expert_class: Expert class to use
            expert_kwargs: Additional arguments for expert initialization
            k: Number of experts to route to
            capacity_factor: Expert capacity multiplier
            router_noise_epsilon: Noise factor for router load balancing
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        
        # Initialize experts
        expert_kwargs = expert_kwargs or {}
        self.experts = nn.ModuleList([
            expert_class(input_dim=input_dim, output_dim=output_dim, **expert_kwargs)
            for _ in range(num_experts)
        ])
        
        # Initialize router
        self.router = Router(
            input_dim=input_dim,
            num_experts=num_experts,
            k=k,
            capacity_factor=capacity_factor,
            noise_epsilon=router_noise_epsilon
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the MoE model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of:
            - output: Model output of shape [batch_size, output_dim]
            - aux_loss: Auxiliary load balancing loss (None if not training)
        """
        batch_size = x.shape[0]
        
        # Get routing weights and indices
        routing_weights, routing_indices, aux_loss = self.router(x)
        
        # Initialize output tensor
        combined_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Compute expert outputs and combine them
        for i in range(self.k):
            expert_indices = routing_indices[:, i]
            expert_weights = routing_weights[:, i].unsqueeze(1)
            
            # Gather expert outputs
            expert_outputs = torch.stack([
                self.experts[idx](x[b:b+1])
                for b, idx in enumerate(expert_indices)
            ])
            
            # Combine expert outputs weighted by routing weights
            combined_output += (expert_outputs.squeeze(1) * expert_weights)
            
        return combined_output, aux_loss
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'k': self.k,
            'expert_class': self.experts[0].__class__.__name__,
            'expert_config': self.experts[0].get_config()
        } 