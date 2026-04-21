import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

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
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Tuple of:
            - output: Model output of shape [..., output_dim]
            - aux_loss: Auxiliary load balancing loss (None if not training)
        """
        # Get routing weights and indices
        leading_shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        routing_weights, routing_indices, aux_loss = self.router(flat_x)

        combined_output = torch.zeros(
            flat_x.shape[0],
            self.output_dim,
            device=flat_x.device,
            dtype=flat_x.dtype,
        )

        token_indices = torch.arange(flat_x.shape[0], device=flat_x.device).repeat_interleave(self.k)
        flat_assignments = routing_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1)

        for expert_idx in flat_assignments.unique(sorted=True).tolist():
            selected = flat_assignments == expert_idx
            selected_tokens = token_indices[selected]
            selected_inputs = flat_x.index_select(0, selected_tokens)
            selected_outputs = self.experts[int(expert_idx)](selected_inputs)
            weighted_outputs = selected_outputs * flat_weights[selected].unsqueeze(-1).to(selected_outputs.dtype)
            combined_output.index_add_(0, selected_tokens, weighted_outputs)

        return combined_output.reshape(*leading_shape, self.output_dim), aux_loss
    
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
