"""
Example showing how to implement a custom evaluator that must satisfy constraints using the Evaluator abstract class.
"""
import torch

from constraints.problems import ConstraintProblem


class BNH(ConstraintProblem):
    """
    Evaluator implementation of the BNH problem. Minimizes 2 objectives f1 and f2 and has 4 constraints although
    we split them into 6 for ease of implementation.
    """
    def __init__(self, batch_size: int = 64, device: str = "mps", n_jobs: int = 1):
        contexts = torch.tensor([[0, 5, 0, 3]], dtype=torch.float32)
        super().__init__(contexts, batch_size=batch_size, device=device, outcomes=["f1", "f2"], n_jobs=n_jobs)

    # pylint:disable=missing-function-docstring
    # pylint: disable=invalid-name
    def f1(self, X: torch.Tensor) -> torch.Tensor:
        """
        f1 = 4x1^2 + 4x2^2
        """
        f = 4 * X ** 2
        f = torch.sum(f, dim=1)
        return f.unsqueeze(1)

    def f2(self, X: torch.Tensor) -> torch.Tensor:
        """
        f2 = (x1 - 5)^2 + (x2 - 5)^2
        """
        f = (X - 5)**2
        f = torch.sum(f, dim=1)
        return f.unsqueeze(1)

    def g1(self, X: torch.Tensor) -> torch.Tensor:
        """
        g1 = (x1-5)^2 + x2^2 <= 25
        X: N x 2
        g: N x 1
        """
        c = torch.tensor([[-5, 0]], dtype=X.dtype, device=X.device)
        g = (c + X) ** 2
        g = torch.sum(g, dim=1)
        g = g - 25
        return g.unsqueeze(1)

    def g2(self, X: torch.Tensor) -> torch.Tensor:
        """
        g2 = (x1 - 8)^2 + (x2 + 3)^2 >= 7.7
        X: N x 2
        g: N x 1
        """
        c = torch.tensor([[-8, 3]], dtype=X.dtype, device=X.device)
        g = (c + X)**2
        g = torch.sum(g, dim=1)
        g = 7.7 - g
        return g.unsqueeze(1)

    def g3(self, X: torch.Tensor) -> torch.Tensor:
        """
        g3 = 0 <= x1 <= 5
        X: N x 2
        g: N x 1
        """
        x1 = X[:, 0]
        lower = torch.where(-1 * x1 < 0, -1 * x1, 0)
        upper = torch.where(x1 - 5 < 0, x1 - 5, 0)
        g = lower + upper
        return g.unsqueeze(1)

    def g4(self, X: torch.Tensor) -> torch.Tensor:
        """
        g4 = 0 <= x2 <= 3
        X: N x 2
        g: N x 1
        """
        x2 = X[:, 1]
        lower = torch.where(-1 * x2 < 0, -1 * x2, 0)
        upper = torch.where(x2 - 3 < 0, x2 - 3, 0)
        g = lower + upper
        return g.unsqueeze(1)
    # pylint:enable=missing-function-docstring

    def compute_outcomes(self, _, actions: torch.Tensor) -> torch.Tensor:
        f1 = self.f1(actions)
        f2 = self.f2(actions)

        f = torch.cat([f1, f2], dim=1)
        return f

    def compute_constraints(self, _, actions: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(actions)
        g2 = self.g2(actions)
        g3 = self.g3(actions)
        g4 = self.g4(actions)

        g = torch.cat([g1, g2, g3, g4], dim=1)
        return g

    def get_optimal(self, n_points: int = 100) -> torch.Tensor:
        """
        x1 = x2 while 0 <= x1 <= 3, then x2 = 3 while 3 <= x1 <= 5
        """
        x1 = torch.linspace(0, 5, n_points)
        x2 = torch.where(x1 <= 3, x1, 3)
        X = torch.stack([x1, x2], dim=1)
        return X
    # pylint:enable=invalid-name
