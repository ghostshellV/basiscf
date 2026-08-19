from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Optional, Tuple
from src.counterfactuals.core import TargetSpec, TSFeatureSchema, LossWeights
# class CounterfactualExplainer(ABC):
#     def __init__(self, model, **kwargs):
#         self.model = model

#     @abstractmethod
#     def generate(self, query_instance: torch.Tensor, target_rul: float) -> torch.Tensor:
#         """
#         Args:
#             query_instance: Input time series (T, D)
#             target_rul: Desired outcome
#         Returns:
#             cf_instance: Counterfactual time series (T, D)
#         """
#         pass

class CounterfactualExplainer(ABC):
    def __init__(self, model, **kwargs):
        self.model = model

    @abstractmethod
    def generate(
        self,
        query_instance: torch.Tensor,
        target: TargetSpec,
        schema: Optional[TSFeatureSchema] = None,
        num_cfs: int = 1,
        loss_weights: Optional[LossWeights] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            query_instance: Input time series of shape (T, D)
            target: Shared target specification
            schema: Shared feature schema
            num_cfs: Number of counterfactuals to generate
            loss_weights: Shared loss-weight container
            verbose: Print optimisation progress

        Returns:
            cfs: Tensor of shape (num_cfs, T, D)
            info: Dictionary with history and summary metrics
        """
        raise NotImplementedError