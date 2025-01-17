import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Callable
import logging
from tqdm import tqdm
import os

from ..models.moe import MixtureOfExperts

class MoETrainer:
    """Trainer class for Mixture of Experts model."""
    
    def __init__(self,
                 model: MixtureOfExperts,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 aux_loss_weight: float = 0.1,
                 gradient_clip_val: Optional[float] = 1.0,
                 logger: Optional[logging.Logger] = None,
                 tensorboard_dir: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: MoE model to train
            criterion: Loss function for the main task
            optimizer: Optimizer for training
            device: Device to train on
            aux_loss_weight: Weight for auxiliary load balancing loss
            gradient_clip_val: Value for gradient clipping (None to disable)
            logger: Logger for training information
            tensorboard_dir: Directory for tensorboard logs
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.gradient_clip_val = gradient_clip_val
        
        # Setup logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup tensorboard
        self.writer = None
        if tensorboard_dir:
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
    
    def train_epoch(self, 
                    train_loader: DataLoader,
                    epoch: int,
                    metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = metrics or {}
        
        total_loss = 0.0
        total_aux_loss = 0.0
        metric_values = {name: 0.0 for name in metrics}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, aux_loss = self.model(data)
            
            # Compute losses
            task_loss = self.criterion(output, target)
            aux_loss = aux_loss if aux_loss is not None else 0.0
            total_loss = task_loss + self.aux_loss_weight * aux_loss
            
            # Backward pass
            total_loss.backward()
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_val
                )
            self.optimizer.step()
            
            # Update metrics
            batch_size = data.size(0)
            total_loss += total_loss.item() * batch_size
            total_aux_loss += aux_loss * batch_size if aux_loss != 0.0 else 0.0
            
            for name, metric_fn in metrics.items():
                metric_values[name] += metric_fn(output, target) * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / ((batch_idx + 1) * batch_size),
                'aux_loss': total_aux_loss / ((batch_idx + 1) * batch_size)
            })
        
        # Compute epoch metrics
        num_samples = len(train_loader.dataset)
        epoch_metrics = {
            'loss': total_loss / num_samples,
            'aux_loss': total_aux_loss / num_samples
        }
        epoch_metrics.update({
            name: value / num_samples
            for name, value in metric_values.items()
        })
        
        # Log metrics
        if self.writer is not None:
            for name, value in epoch_metrics.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
        
        return epoch_metrics
    
    def evaluate(self,
                val_loader: DataLoader,
                metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        metrics = metrics or {}
        
        total_loss = 0.0
        total_aux_loss = 0.0
        metric_values = {name: 0.0 for name in metrics}
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, aux_loss = self.model(data)
                
                # Compute losses
                task_loss = self.criterion(output, target)
                aux_loss = aux_loss if aux_loss is not None else 0.0
                
                # Update metrics
                batch_size = data.size(0)
                total_loss += task_loss.item() * batch_size
                total_aux_loss += aux_loss * batch_size if aux_loss != 0.0 else 0.0
                
                for name, metric_fn in metrics.items():
                    metric_values[name] += metric_fn(output, target) * batch_size
        
        # Compute evaluation metrics
        num_samples = len(val_loader.dataset)
        eval_metrics = {
            'loss': total_loss / num_samples,
            'aux_loss': total_aux_loss / num_samples
        }
        eval_metrics.update({
            name: value / num_samples
            for name, value in metric_values.items()
        })
        
        return eval_metrics
    
    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_config(),
            **kwargs
        }
        torch.save(checkpoint, path)
        self.logger.info(f'Saved checkpoint to {path}')
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f'Loaded checkpoint from {path}')
        return checkpoint 