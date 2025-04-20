from typing import Optional
import torch
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)

# conform to the function signature in https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
# `weight` and `size_average` are placeholders and not implemented yet
def liger_cross_entropy_z_loss(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index: int = -100,
    reduce=None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    lse_square_scale: float = 1e-4,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
):
    loss, _ = LigerCrossEntropyFunction.apply(
        input,
        target,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
    )
    return loss

class LigerCrossEntropyLosswithZ(torch.nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        lse_square_scale: float = 1e-4,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (
            label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        assert (label_smoothing >= 0) and (
            label_smoothing <= 1
        ), f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert (
            softcap is None or softcap > 0
        ), f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss

    def forward(self, _input: torch.Tensor, target: torch.Tensor):
        loss, _ = LigerCrossEntropyFunction.apply(
            _input,
            target,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
        )
        return loss