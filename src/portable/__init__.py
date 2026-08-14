"""Capa de inferencia portátil: corre el modelo en NumPy puro, sin MLX.

Pensada para equipos viejos e IoT (Raspberry Pi, x86 antiguo, RK3588, etc.).
Solo depende de: numpy, safetensors, tokenizers.
"""
from .engine import GPTNumpy, cargar

__all__ = ["GPTNumpy", "cargar"]
