"""
Template for constraint problems.
"""
import numpy as np
from presp.evaluator import Evaluator
from presp.prescriptor import NNPrescriptor
import torch
from torch.utils.data import DataLoader, TensorDataset


class ConstraintProblem(Evaluator):
    """
    Base class for constraint problems. One simply needs to implement the compute_outcomes and compute_constraints
    functions to compute the outcomes and constraints for a given set of actions.
    """
    def __init__(self, contexts: torch.Tensor, batch_size: int, device: str, outcomes: list[str], n_jobs: int = 1):
        self.contexts = contexts
        self.batch_size = batch_size
        self.device = device

        super().__init__(outcomes=outcomes, n_jobs=n_jobs)

    def update_predictor(self, _):
        pass

    def compute_outcomes(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes outcome metrics for each set of context/actions.
        contexts: N x C
        actions: N x A
        outcomes: N x O
        """
        raise NotImplementedError("compute_outcomes must be implemented in subclasses")

    def compute_constraints(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes constraint metrics for each set of actions.
        contexts: N x C
        actions: N x A
        constraints: N x G
        """
        raise NotImplementedError("compute_constraints must be implemented in subclasses")

    def prescribe_and_predict(self, candidate: NNPrescriptor,
                              contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs context through a candidate to get its prescribed actions, then predicts outcomes using context/actions.
        Returns the actions, outcomes, and constraints.
        """
        context_dataset = TensorDataset(contexts)
        dataloader = DataLoader(context_dataset, batch_size=self.batch_size, shuffle=False)
        all_actions = []
        all_outcomes = []
        all_constraints = []
        for batch in dataloader:
            # Evaluation process: context -> actions -> outcomes + constraints
            batch = batch[0].to(self.device)
            actions = candidate.forward(batch)
            outcomes = self.compute_outcomes(batch, actions)
            constraints = self.compute_constraints(batch, actions)

            all_actions.append(actions)
            all_outcomes.append(outcomes)
            all_constraints.append(constraints)

        # Consolidate all actions, outcomes, and constraints
        all_actions = torch.cat(all_actions, dim=0)
        all_outcomes = torch.cat(all_outcomes, dim=0)
        all_constraints = torch.cat(all_constraints, dim=0)

        assert all_actions.shape[0] == all_outcomes.shape[0] == all_constraints.shape[0], \
            "Mismatch in number of actions, outcomes, and constraints"

        return all_actions, all_outcomes, all_constraints

    def evaluate_candidate(self, candidate: NNPrescriptor) -> tuple[np.ndarray, float]:
        """
        Evaluates candidate solutions.
        outcomes: N x O, constraints: N x G
        Returns the average outcomes across contexts and the average constraint violation total.
        """
        _, outcomes, constraints = self.prescribe_and_predict(candidate, self.contexts)
        # Average across context datapoints
        outcomes = outcomes.mean(dim=0)
        g = torch.where(constraints > 0, constraints, 0).sum(dim=1).mean(dim=0).item()

        return outcomes.cpu().numpy(), g
