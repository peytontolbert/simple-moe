# Simple MoE: A PyTorch Mixture of Experts Implementation

A clean, modular implementation of Mixture of Experts (MoE) models in PyTorch. This repository provides a flexible framework for building, training, and evaluating MoE models.

## What is a Mixture of Experts (MoE)?

Mixture of Experts is a machine learning architecture that combines multiple "expert" neural networks, each specializing in different aspects of a task, with a trainable routing mechanism that decides which experts to use for each input.

Key components:
1. **Experts**: Specialized neural networks that each focus on different parts of the input space
2. **Router**: A learned mechanism that determines which experts should process each input
3. **Combination**: A method to combine the outputs of the selected experts

Benefits of MoE:
- Increased model capacity without proportional increase in computation
- Specialization of experts in different aspects of the task
- Efficient processing by only using relevant experts for each input

## Installation

```bash
git clone https://github.com/peytontolbert/simple-moe.git
cd simple-moe
pip install -r requirements.txt
```

## Quick Start

Here's a simple example of creating and training an MoE model:

```python
import torch
from src.models.moe import MixtureOfExperts
from src.training.trainer import MoETrainer

# Create MoE model
model = MixtureOfExperts(
    input_dim=128,
    output_dim=10,
    num_experts=4,
    k=2  # number of experts to route to
)

# Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Initialize trainer
trainer = MoETrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    aux_loss_weight=0.1  # weight for load balancing loss
)

# Train model
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(train_loader, epoch)
    eval_metrics = trainer.evaluate(val_loader)
```

## Features

- **Modular Architecture**:
  - Customizable expert networks
  - Configurable routing mechanisms
  - Flexible training strategies

- **Training Utilities**:
  - Integrated TensorBoard logging
  - Checkpoint management
  - Custom metrics tracking

- **Evaluation Metrics**:
  - Expert utilization analysis
  - Routing pattern visualization
  - Performance benchmarking

## Repository Structure

```
simple-moe/
├── src/
│   ├── models/
│   │   ├── moe.py        # Core MoE implementation
│   │   ├── experts.py    # Expert network implementations
│   │   └── router.py     # Routing mechanism
│   ├── training/
│   │   └── trainer.py    # Training utilities
│   └── evaluation/
│       └── metrics.py    # Evaluation metrics
```

## Advanced Usage

### Custom Expert Networks

Create custom expert networks by inheriting from `ExpertBase`:

```python
from src.models.experts import ExpertBase

class CustomExpert(ExpertBase):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        # Custom expert architecture
```

### Router Configuration

Configure routing behavior:

```python
model = MixtureOfExperts(
    input_dim=128,
    output_dim=10,
    num_experts=4,
    k=2,
    capacity_factor=1.2,
    router_noise_epsilon=1e-2
)
```

### Training Configuration

Customize training behavior:

```python
trainer = MoETrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    aux_loss_weight=0.1,
    gradient_clip_val=1.0,
    tensorboard_dir="runs/experiment1"
)
```

## Monitoring and Evaluation

### Training Metrics
- Task-specific loss
- Load balancing loss
- Expert utilization
- Routing patterns

### Evaluation Metrics
- Expert specialization scores
- Routing distribution analysis
- Capacity utilization
- Inter-expert correlation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
