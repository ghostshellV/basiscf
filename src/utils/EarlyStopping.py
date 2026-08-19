# import torch
# import numpy as np
# import os

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0, path='', start_epoch=1):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                            Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                         Default: 'saved_model/checkpoint.pt'
#             start_epoch (int): Epoch number (1-based) from which to start monitoring for early stopping.
#                                Saving the best model will still occur before this epoch.
#                                Default: 1 (start immediately)
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.delta = delta
#         self.path = path
#         self.start_epoch = start_epoch # Store the epoch to start checking
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#         # Ensure the directory for the path exists during initialization
#         if self.path:
#             os.makedirs(os.path.dirname(self.path), exist_ok=True)


#     def __call__(self, val_loss, model, epoch): # Add epoch parameter
#         """
#         Args:
#             val_loss (float): Validation loss for the current epoch.
#             model (torch.nn.Module): Model to save if validation loss improves.
#             epoch (int): The current epoch number (1-based).
#         """
#         score = -val_loss

#         # --- Check for improvement and save best model ALWAYS ---
#         if self.best_score is None:
#             # First call or first improvement
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0 # Reset counter on improvement
#         elif score > self.best_score + self.delta:
#             # Significant improvement detected
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0 # Reset counter on improvement
#         else:
#             # --- No improvement ---
#             # Only increment counter and check for stopping *after* start_epoch
#             if epoch >= self.start_epoch:
#                 self.counter += 1
#                 if self.verbose:
#                     print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Epoch {epoch} >= Start Epoch {self.start_epoch})')
#                 if self.counter >= self.patience:
#                     self.early_stop = True
#             # Optional: Print message if before start_epoch and verbose
#             elif self.verbose:
#                  print(f'Epoch {epoch}: Early stopping checks deferred until epoch {self.start_epoch}. Loss did not improve.')

#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decreases.'''
#         if self.verbose:
#             # Ensure val_loss_min is updated *before* printing if it's the first save
#             old_min = self.val_loss_min
#             print(f'Validation loss decreased ({old_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
#         # Ensure the directory exists before saving (redundant if checked in init, but safe)
#         os.makedirs(os.path.dirname(self.path), exist_ok=True)
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss # Update the minimum loss

#     def load_checkpoint(self, model):
#         '''Loads the saved model state dictionary.'''
#         if os.path.exists(self.path):
#             model.load_state_dict(torch.load(self.path))
#             print(f"Loaded checkpoint from '{self.path}'")
#         else:
#             print(f"Checkpoint file not found at '{self.path}', model weights not loaded.")

import os
import numpy as np
import torch
from typing import Callable, Optional


class EarlyStopping:
    """
    Early stopping with:
      - configurable metric direction (mode='min' or 'max')
      - best_epoch tracking
      - optional save callback for consistent checkpointing
    """

    def __init__(
        self,
        patience: int = 20,
        verbose: bool = False,
        delta: float = 0.0,
        start_epoch: int = 1,
        mode: str = "min",
        path: str = "",
        save_fn: Optional[Callable[[int, bool], None]] = None,
    ):
        """
        Args:
            patience: epochs to wait after no improvement (after start_epoch).
            verbose: prints diagnostics.
            delta: minimum change to qualify as improvement.
            start_epoch: begin counting patience only after this epoch (1-based).
            mode: 'min' (lower is better) or 'max' (higher is better).
            path: if save_fn is None, will save best model.state_dict() here.
            save_fn: optional callback called on improvement as save_fn(epoch, is_best=True).
                     This is the recommended way so you keep ONE saving format.
        """
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"

        self.patience = patience
        self.verbose = verbose
        self.delta = float(delta)
        self.start_epoch = int(start_epoch)
        self.mode = mode

        self.path = path
        self.save_fn = save_fn

        self.counter = 0
        self.early_stop = False

        self.best_metric = None
        self.best_epoch = None

        # prepare directory if we will use path saving
        if self.path and self.save_fn is None:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)

    def _is_improvement(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < (self.best_metric - self.delta)
        else:
            return metric > (self.best_metric + self.delta)

    def __call__(self, metric_value: float, model: torch.nn.Module, epoch: int):
        """
        Args:
            metric_value: monitored value (e.g., val_loss or nasa_score)
            model: model to save (only used if save_fn is None and path is provided)
            epoch: current epoch (1-based)
        """
        metric_value = float(metric_value)

        improved = self._is_improvement(metric_value)

        if improved:
            self.best_metric = metric_value
            self.best_epoch = int(epoch)
            self.counter = 0

            if self.verbose:
                print(f"[EarlyStopping] Improvement at epoch {epoch}: best_metric={self.best_metric:.6f}")

            # Save best model
            if self.save_fn is not None:
                self.save_fn(epoch, True)  # your Trainer.save_checkpoint wrapper
            elif self.path:
                self._save_state_dict(model)

        else:
            # only start counting patience after start_epoch
            if epoch >= self.start_epoch:
                self.counter += 1
                if self.verbose:
                    print(f"[EarlyStopping] no improve: {self.counter}/{self.patience} (epoch {epoch})")
                if self.counter >= self.patience:
                    self.early_stop = True
            elif self.verbose:
                print(f"[EarlyStopping] epoch {epoch}: checks deferred until epoch {self.start_epoch}")

    def _save_state_dict(self, model: torch.nn.Module):
        # safe directory creation
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"[EarlyStopping] saved best state_dict -> {self.path}")

    def load_checkpoint(self, model: torch.nn.Module):
        """
        Load best weights saved via 'path' mode.
        (If you're using save_fn, load from whatever file you saved there.)
        """
        if not self.path:
            raise ValueError("EarlyStopping.path is empty; cannot load checkpoint.")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")
        model.load_state_dict(torch.load(self.path, map_location="cpu"))
        if self.verbose:
            print(f"[EarlyStopping] loaded best state_dict <- {self.path}")
