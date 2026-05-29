from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermvander
from numpy.polynomial.laguerre import lagvander
from numpy.polynomial.polynomial import polyvander


def design_matrix(x: np.ndarray, family: str = "laguerre", degree: int = 3) -> np.ndarray:
    """Construct regression basis matrix for LSMC continuation value."""

    scaled = x / max(float(np.mean(x)), 1e-12)
    if family == "polynomial":
        return polyvander(scaled, degree)
    if family == "laguerre":
        return lagvander(scaled, degree)
    if family == "hermite":
        return hermvander(scaled, degree)
    raise ValueError(f"unsupported basis family: {family}")

