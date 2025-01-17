import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

def expert_utilization(routing_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute expert utilization from routing weights.
    
    Args:
        routing_weights: Tensor of shape [batch_size, num_experts]
        
    Returns:
        Tensor of shape [num_experts] containing utilization per expert
    """
    return routing_weights.mean(dim=0)

def expert_capacity_utilization(routing_weights: torch.Tensor,
                              capacity: int) -> torch.Tensor:
    """
    Compute expert capacity utilization.
    
    Args:
        routing_weights: Tensor of shape [batch_size, num_experts]
        capacity: Maximum capacity per expert
        
    Returns:
        Tensor of shape [num_experts] containing capacity utilization per expert
    """
    expert_counts = routing_weights.sum(dim=0)
    return expert_counts / capacity

def routing_entropy(routing_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of routing distribution.
    
    Args:
        routing_weights: Tensor of shape [batch_size, num_experts]
        
    Returns:
        Scalar tensor containing routing entropy
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs = routing_weights + eps
    return -(probs * torch.log(probs)).sum(dim=-1).mean()

def expert_correlation(expert_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute correlation between expert outputs.
    
    Args:
        expert_outputs: List of tensors of shape [batch_size, output_dim]
        
    Returns:
        Tensor of shape [num_experts, num_experts] containing pairwise correlations
    """
    num_experts = len(expert_outputs)
    correlations = torch.zeros(num_experts, num_experts)
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            corr = F.cosine_similarity(
                expert_outputs[i].flatten(1),
                expert_outputs[j].flatten(1)
            ).mean()
            correlations[i, j] = corr
            correlations[j, i] = corr
            
    return correlations

class MoEMetrics:
    """Collection of metrics for evaluating MoE models."""
    
    def __init__(self, num_experts: int, expert_capacity: Optional[int] = None):
        """
        Initialize MoE metrics.
        
        Args:
            num_experts: Number of experts in the model
            expert_capacity: Maximum capacity per expert (if using capacity constraints)
        """
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
    def compute_metrics(self,
                       routing_weights: torch.Tensor,
                       expert_outputs: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Compute all MoE metrics.
        
        Args:
            routing_weights: Tensor of shape [batch_size, num_experts]
            expert_outputs: Optional list of expert outputs for correlation analysis
            
        Returns:
            Dictionary containing computed metrics
        """
        metrics = {
            'expert_utilization': expert_utilization(routing_weights),
            'routing_entropy': routing_entropy(routing_weights)
        }
        
        if self.expert_capacity is not None:
            metrics['capacity_utilization'] = expert_capacity_utilization(
                routing_weights, self.expert_capacity
            )
            
        if expert_outputs is not None:
            metrics['expert_correlation'] = expert_correlation(expert_outputs)
            
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()} 