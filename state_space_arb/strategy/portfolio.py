"""Two-leg portfolio with explicit commission + slippage accounting."""
from __future__ import annotations

from ..config.settings import CostModel
from .signals import Position


class Portfolio:
    def __init__(self, cash: float, costs: CostModel) -> None:
        self.cash = float(cash)
        self.costs = costs
        self.qty_y = 0
        self.qty_x = 0
        self.position = Position.FLAT
        self.total_costs = 0.0

    def equity(self, py: float, px: float) -> float:
        return self.cash + self.qty_y * py + self.qty_x * px

    def _fill(self, shares: int, price: float) -> None:
        if shares == 0:
            return
        side = 1 if shares > 0 else -1
        fill = price + side * self.costs.slippage_per_share          # slippage always hurts
        fee = self.costs.commission(abs(shares), fill)
        self.cash -= shares * fill + fee
        self.total_costs += abs(shares) * self.costs.slippage_per_share + fee

    def open_spread(self, direction: Position, py: float, px: float, beta: float, allocation: float) -> bool:
        """Size so the reference leg hedges the target leg in shares: qty_x = beta * qty_y.

        (y ~ beta*x, so one share of y is offset by beta shares of x.)  Returns False if
        the order can't be sized (zero shares or non-positive beta).
        """
        if direction == Position.FLAT or beta <= 0:
            return False
        qty_y = int(self.equity(py, px) * allocation // (py + self.costs.slippage_per_share))
        qty_x = int(round(beta * qty_y))
        if qty_y <= 0 or qty_x <= 0:
            return False
        sign = int(direction)                     # +1 long spread: buy y, sell x
        self._fill(sign * qty_y, py)
        self._fill(-sign * qty_x, px)
        self.qty_y, self.qty_x = sign * qty_y, -sign * qty_x
        self.position = direction
        return True

    def close_spread(self, py: float, px: float) -> None:
        self._fill(-self.qty_y, py)
        self._fill(-self.qty_x, px)
        self.qty_y = self.qty_x = 0
        self.position = Position.FLAT
