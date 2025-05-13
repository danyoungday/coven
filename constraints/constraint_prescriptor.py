"""
Prescriptor for hard constraints.
"""

from presp.prescriptor import NNPrescriptor
import torch


class ConstraintPrescriptor(NNPrescriptor):
    """
    Hard constraint prescriptor. Takes the inputs to the model assuming they are in pairs of (min, max) values and then
    scales the output of the model to be within those bounds.
    """
    def __init__(self, model_params: list[dict], device: str = "cpu"):
        if model_params[-1]["type"] != "sigmoid":
            raise ValueError("The last layer of a hard-constrained model must be sigmoid.")
        if model_params[0]["in_features"] != model_params[-2]["out_features"] * 2:
            raise ValueError("There should be twice as many inputs as outputs.")
        super().__init__(model_params=model_params, device=device)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Applies the model to the input then scales the output by input.
        """
        with torch.no_grad():
            output = super().forward(context)
            # Take context and stack into (N, 2, D//2) shape where every other element gets split
            mins = context[:, ::2]
            maxes = context[:, 1::2]
            output = mins + (maxes - mins) * output
            return output
