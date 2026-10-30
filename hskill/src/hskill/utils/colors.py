"""Linear diffuse colors to display sRGB."""

from __future__ import annotations

def srgb(linear: tuple[float, float, float], alpha: float = 1.0) -> tuple[float, ...]:
    def encode(c: float) -> float:
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return (*(encode(c) for c in linear), alpha)
