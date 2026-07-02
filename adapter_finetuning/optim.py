"""Custom learning rate schedulers."""

import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.

    Implementation inspired by PyTorch's CosineAnnealingLR.

    During warmup phase (epochs 0 to warmup_epochs-1):
        - LR increases linearly from warmup_start_lr to base_lr

    After warmup phase (epochs warmup_epochs to max_epochs-1):
        - LR follows cosine annealing from base_lr to eta_min
        - Uses the same formula as torch.optim.lr_scheduler.CosineAnnealingLR

    The cosine annealing formula after warmup is:
        eta_t = eta_min + (base_lr - eta_min) * (1 + cos(pi * T_cur / T_max)) / 2
    where T_cur is the number of epochs since warmup ended, and
    T_max is the total number of epochs after warmup (max_epochs - warmup_epochs).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of epochs for the warmup phase.
    max_epochs : int
        Total number of training epochs (including warmup).
    warmup_start_lr : float, optional
        Learning rate at the start of warmup. Default: 0.
    eta_min : float, optional
        Minimum learning rate at the end of cosine annealing. Default: 0.
    last_epoch : int, optional
        The index of last epoch. Default: -1.
        When resuming training, set this to the last completed epoch.

    Example
    -------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    >>> scheduler = CosineAnnealingWarmupLR(
    ...     optimizer, warmup_epochs=5, max_epochs=100, warmup_start_lr=1e-6
    ... )
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        # T_max for cosine annealing is the number of epochs after warmup
        self.T_max = max_epochs - warmup_epochs

        # Initialize parent (this will call get_lr() once)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for the current epoch.

        Note: last_epoch represents the current position in the schedule.
        After _initial_step(), last_epoch starts at 0.
        - During warmup: last_epoch goes from 0 to (warmup_epochs - 1)
        - After warmup: last_epoch goes from warmup_epochs to (max_epochs - 1)
        """
        # Handle initial step: return warmup_start_lr for all groups
        if self._is_initial:
            return [self.warmup_start_lr for _ in self.optimizer.param_groups]

        # Warmup phase: linear interpolation
        if self.last_epoch < self.warmup_epochs:
            # Use closed-form formula for warmup
            warmup_progress = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]

        # Cosine annealing phase (follows CosineAnnealingLR pattern)
        else:
            T_cur = self.last_epoch - self.warmup_epochs

            # First cosine step (right after warmup): return base_lr
            if T_cur == 0:
                return [base_lr for base_lr in self.base_lrs]

            # Use closed-form for cosine annealing
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * T_cur / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]

    def _get_closed_form_lr(self):
        """Compute learning rate using closed-form formula.

        This is useful for debugging and provides the exact LR
        without relying on recursive updates.
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            T_cur = self.last_epoch - self.warmup_epochs
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * T_cur / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]

if __name__=="__main__":
    # Simple test to verify scheduler behavior
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_epochs=5,
        max_epochs=50,
        warmup_start_lr=1e-6,
        eta_min=1e-6,
    )

    lrs = []
    epochs = []

    # Get initial LR (before any training)
    initial_lr = optimizer.param_groups[0]["lr"]
    lrs.append(initial_lr)
    epochs.append(0)
    print(f"Epoch 0 (initial): last_epoch={scheduler.last_epoch}, LR={initial_lr:.6e}")

    for epoch in range(1, 50):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        epochs.append(epoch)
        # if epoch < 3:
        print(f"Epoch {epoch}: last_epoch={scheduler.last_epoch}, LR={current_lr:.6e}")

    plt.plot(epochs, lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("CosineAnnealingWarmupLR Schedule")
    plt.show()