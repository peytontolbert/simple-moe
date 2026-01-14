from flax import linen as nn
from jax import numpy as jnp
from jax import lax
from jax.nn import softmax
from jax import random
from typing import Tuple, Optional

class Router(nn.Module):
    """Router network for Mixture of Experts model."""
    input_dim: int
    num_experts: int
    k: int = 1
    capacity_factor: float = 1.0
    noise_epsilon: float = 1e-2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Route input to top-k experts.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            training: Whether the model is in training mode
            
        Returns:
            Tuple of:
            - routing_weights: Tensor of shape [batch_size, k]
            - routing_indices: Tensor of shape [batch_size, k]
            - aux_loss: Load balancing auxiliary loss (None if not training)
        """
        batch_size = x.shape[0]
        
        # Router weights
        router = nn.Dense(features=self.num_experts, use_bias=False)
        
        # Add noise during training for better load balancing
        if training:
            noise = random.normal(self.make_rng('dropout'), x.shape) * self.noise_epsilon
            x = x + noise
            
        routing_scores = router(x)
        
        # Get top-k routing weights and indices
        routing_weights, routing_indices = lax.top_k(routing_scores, self.k)
        routing_weights = softmax(routing_weights, axis=-1)
        
        # Compute load balancing loss during training
        aux_loss = None
        if training:
            # Compute expert assignment counts
            expert_counts = jnp.zeros(self.num_experts)
            
            def update_counts(i, counts):
                return counts.at[routing_indices.flatten()[i]].add(1)
            
            expert_counts = lax.fori_loop(0, routing_indices.size, update_counts, expert_counts)
                
            # Compute load balancing loss
            target_count = jnp.ones_like(expert_counts) * (batch_size * self.k / self.num_experts)
            aux_loss = jnp.mean((expert_counts - target_count)**2)
            
        return routing_weights, routing_indices, aux_loss