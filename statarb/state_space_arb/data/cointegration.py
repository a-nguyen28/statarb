"""Engle-Granger cointegration check (statsmodels)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class CointegrationResult:
    adf_statistic: float
    p_value: float
    critical_values: dict[str, float]
    significance: float

    @property
    def is_cointegrated(self) -> bool:
        return self.p_value < self.significance

    @property
    def warning(self) -> str | None:
        if self.is_cointegrated:
            return None
        return (
            f"p-value {self.p_value:.3f} >= {self.significance}: relationship is statistically "
            "unreliable for mean reversion."
        )


def test_cointegration(y, x, significance: float = 0.05) -> CointegrationResult:
    stat, p, crit = coint(np.asarray(y, float), np.asarray(x, float), trend="c", autolag="aic")
    return CointegrationResult(
        adf_statistic=float(stat),
        p_value=float(p),
        critical_values={"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        significance=significance,
    )


test_cointegration.__test__ = False  # stop pytest from collecting this as a test
