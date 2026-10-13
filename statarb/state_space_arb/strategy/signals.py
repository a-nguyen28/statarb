"""Signal state machine: z-score in, desired position out. Used identically by backtest and live."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from ..config.settings import SignalConfig


class Position(IntEnum):
    SHORT_SPREAD = -1   # short target, long reference
    FLAT = 0
    LONG_SPREAD = 1     # long target, short reference


@dataclass(frozen=True)
class SignalEvent:
    position: Position   # desired position AFTER seeing this bar
    reason: str          # hold | entry | exit | stop | timeout | cooldown


class SignalStateMachine:
    """spread = y - (beta*x + alpha).  z < -entry  => y is cheap => LONG_SPREAD.

    Stop logic: if the spread moves against you past `stop_z` for `stop_bars` consecutive bars,
    that's treated as a structural break: flatten and sit out `cooldown_bars`.
    """

    def __init__(self, cfg: SignalConfig) -> None:
        self.cfg = cfg
        self.position = Position.FLAT
        self.structural_breaks = 0
        self._beyond = 0
        self._held = 0
        self._cooldown = 0

    def _go_flat(self, reason: str) -> SignalEvent:
        self.position = Position.FLAT
        self._beyond = self._held = 0
        return SignalEvent(Position.FLAT, reason)

    def force_stop(self, reason: str = "loss_stop") -> SignalEvent:
        """External hard stop (dollar loss limit): flatten, count a structural break, start cooldown."""
        self.structural_breaks += 1
        self._cooldown = self.cfg.cooldown_bars
        return self._go_flat(reason)

    def step(self, z: float) -> SignalEvent:
        c = self.cfg
        if not math.isfinite(z):
            return SignalEvent(self.position, "hold")

        if self._cooldown > 0:
            self._cooldown -= 1
            return SignalEvent(Position.FLAT, "cooldown")

        if self.position == Position.FLAT:
            if c.entry_z <= abs(z) < c.stop_z:
                self.position = Position.LONG_SPREAD if z < 0 else Position.SHORT_SPREAD
                self._beyond = self._held = 0
                return SignalEvent(self.position, "entry")
            return SignalEvent(Position.FLAT, "hold")

        # in a position
        self._held += 1
        adverse = -z if self.position == Position.LONG_SPREAD else z   # positive = moving against us
        self._beyond = self._beyond + 1 if adverse >= c.stop_z else 0
        if self._beyond >= c.stop_bars:
            self.structural_breaks += 1
            self._cooldown = c.cooldown_bars
            return self._go_flat("stop")
        if c.max_holding_bars is not None and self._held >= c.max_holding_bars:
            return self._go_flat("timeout")
        reverted = z >= c.exit_z if self.position == Position.LONG_SPREAD else z <= -c.exit_z
        if reverted:
            return self._go_flat("exit")
        return SignalEvent(self.position, "hold")
