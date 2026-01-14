from flax import linen as nn
import jax
from jax import numpy as jnp
from typing import Optional, Tuple

from experts_jax import FFNExpert
from router_jax import Router

class MixtureOfExperts(nn.Module):
    """Mixture of Experts model implementation."""
    input_dim: int
    output_dim: int
    num_experts: int
    expert_class: nn.Module = FFNExpert
    expert_kwargs: Optional[dict] = None
    k: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass through the MoE model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            training: Whether the model is in training mode
            
        Returns:
            Tuple of:
            - output: Model output of shape [batch_size, output_dim]
            - aux_loss: Auxiliary load balancing loss (None if not training)
        """
        
        # Initialize router
        router = Router(
            input_dim=self.input_dim,
            num_experts=self.num_experts,
            k=self.k
        )
        
        # Get routing weights and indices
        routing_weights, routing_indices, aux_loss = router(x, training)
        
        # Initialize experts
        expert_kwargs = self.expert_kwargs or {}
        experts = [self.expert_class(input_dim=self.input_dim, output_dim=self.output_dim, **expert_kwargs) for _ in range(self.num_experts)]

        # Get expert outputs
        expert_outputs = jnp.stack([expert(x, training) for expert in experts], axis=1)
        
        # Select the top-k expert outputs for each input
        one_hot_indices = jax.nn.one_hot(routing_indices, num_classes=self.num_experts)
        selected_expert_outputs = jnp.einsum('bkn,bno->bko', one_hot_indices, expert_outputs)
        
        # Weight the selected expert outputs by the routing weights
        weighted_expert_outputs = jnp.einsum('bk,bko->bo', routing_weights, selected_expert_outputs)
        
        return weighted_expert_outputs, aux_loss

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'k': self.k,
            'expert_class': self.expert_class.__name__,
            'expert_config': self.expert_class(input_dim=self.input_dim, output_dim=self.output_dim, **(self.expert_kwargs or {})).get_config()
        }