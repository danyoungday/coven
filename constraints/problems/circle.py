import torch

from constraints.problems import ConstraintProblem


class Circle(ConstraintProblem):
    """
    Circle problem. We minimize x1 and x2 subject to staying outside the circle radius c and greater than 0.
    """
    def __init__(self):
        contexts = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32)
        # test = torch.tensor([[0.5], [3.0], [5.0]], dtype=torch.float32)
        super().__init__(contexts, batch_size=64, device="mps", outcomes=["f1", "f2"], n_jobs=-1)

    def get_optimal_actions(self, contexts: torch.Tensor) -> torch.Tensor:
        """
        Returns the optimal actions for each context.
        returns X: c x a x L where L is the number of points we want to use to plot.
        """
        X = []
        for c in contexts:
            c = c.item()
            x1 = torch.linspace(0, c, steps=1000, device=self.device)
            x2 = torch.sqrt(c**2 - x1**2)
            x = torch.stack([x1, x2], dim=0)
            X.append(x)

        X = torch.stack(X, dim=0)
        return X

    # pylint: disable=invalid-name
    def f1(self, X: torch.Tensor) -> float:
        """
        Minimize x1
        """
        return X[:, 0].unsqueeze(1)

    def f2(self, X: torch.Tensor) -> float:
        """
        Minimize x2
        """
        return X[:, 1].unsqueeze(1)

    def g1(self, X: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """
        X: 3x2 -> 3x1
        Constraint that forces x1 and x2 to be outside of the circle radius c.
        x1^2 + x2^2 - c >= 0
        """
        return -1 * (torch.sum(X**2, dim=1).unsqueeze(1) - contexts**2)

    def g2(self, X: torch.Tensor) -> torch.Tensor:
        """
        Constraint forcing x1 to be greater than 0
        """
        return -1 * X[:, 0].unsqueeze(1)

    def g3(self, X: torch.Tensor) -> torch.Tensor:
        """
        Constraint forcing x2 to be greater than 0
        """
        return -1 * X[:, 1].unsqueeze(1)

    def compute_outcomes(self, _, actions: torch.Tensor) -> torch.Tensor:
        # We're just returning actions here, but we may need the f1 and f2 functions later.
        f1 = self.f1(actions)
        f2 = self.f2(actions)
        return torch.cat([f1, f2], dim=1)

    def compute_constraints(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        g1 = self.g1(actions, contexts)
        g2 = self.g2(actions)
        g3 = self.g3(actions)
        return torch.cat([g1, g2, g3], dim=1)
