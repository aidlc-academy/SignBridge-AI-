"""
Pure-NumPy inference for a small Keras ``Sequential`` LSTM classifier.

Why NumPy: the SignBridge backend must not depend on TensorFlow (it would force
``ml-dtypes`` 0.2.0, which conflicts with the MediaPipe/jax stack installed
here). The model is tiny (3 LSTM layers + 3 Dense layers, ~0.6M parameters),
so a direct forward pass is fast and easy to verify against TensorFlow.

This module is an original implementation of the standard Keras LSTM equations
(gate order i, f, c, o). It only *reads* a weights file; it contains no code
from the model's source repository. The weights file is NOT part of this
project and must be supplied by configuration (see ``LSTM_WEIGHTS_PATH``).

Supported layers (anything else raises ``UnsupportedModelError`` rather than
silently producing wrong numbers):
  InputLayer, LSTM(activation=relu|tanh, recurrent_activation=sigmoid),
  Dense(activation=relu|softmax|linear).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class UnsupportedModelError(RuntimeError):
    """The weights file describes an architecture this runtime does not implement."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": _relu,
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "linear": lambda x: x,
    "softmax": _softmax,
}


@dataclass
class _Lstm:
    name: str
    kernel: np.ndarray            # (in, 4*units)
    recurrent_kernel: np.ndarray  # (units, 4*units)
    bias: np.ndarray              # (4*units,)
    units: int
    activation: Callable[[np.ndarray], np.ndarray]
    return_sequences: bool

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (T, in) -> (T, units) if return_sequences else (units,)."""
        steps = x.shape[0]
        u = self.units
        # Input projection for every timestep at once.
        proj = x @ self.kernel + self.bias            # (T, 4u)
        h = np.zeros(u, dtype=np.float32)
        c = np.zeros(u, dtype=np.float32)
        outs = np.empty((steps, u), dtype=np.float32) if self.return_sequences else None
        for t in range(steps):
            z = proj[t] + h @ self.recurrent_kernel   # (4u,)
            i = _sigmoid(z[0:u])
            f = _sigmoid(z[u:2 * u])
            g = self.activation(z[2 * u:3 * u])
            o = _sigmoid(z[3 * u:4 * u])
            c = f * c + i * g
            h = (o * self.activation(c)).astype(np.float32)
            c = c.astype(np.float32)
            if outs is not None:
                outs[t] = h
        return outs if outs is not None else h


@dataclass
class _Dense:
    name: str
    kernel: np.ndarray
    bias: np.ndarray
    activation: Callable[[np.ndarray], np.ndarray]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x @ self.kernel + self.bias).astype(np.float32)


def _attr_text(v) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


class LstmSequenceClassifier:
    """Loads a Keras-2 ``Sequential`` LSTM classifier from an HDF5 file."""

    def __init__(self, layers: Sequence, input_shape: Tuple[int, int], n_classes: int, source: Path):
        self._layers = list(layers)
        self.sequence_length, self.feature_size = input_shape
        self.n_classes = n_classes
        self.source = source
        self.param_count = int(
            sum(
                l.kernel.size + l.bias.size + (l.recurrent_kernel.size if isinstance(l, _Lstm) else 0)
                for l in self._layers
            )
        )

    # ------------------------------------------------------------------ load
    @classmethod
    def from_h5(cls, path: Path) -> "LstmSequenceClassifier":
        import h5py  # lazy: only needed when the model is actually loaded

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"LSTM weights file not found: {path}")

        with h5py.File(path, "r") as f:
            if "model_config" not in f.attrs or "model_weights" not in f:
                raise UnsupportedModelError("File is not a full Keras model (missing model_config/model_weights).")
            config = json.loads(_attr_text(f.attrs["model_config"]))
            if config.get("class_name") != "Sequential":
                raise UnsupportedModelError(f"Only Sequential models are supported, got {config.get('class_name')!r}.")
            mw = f["model_weights"]

            layers: List = []
            input_shape: Optional[Tuple[int, int]] = None
            for spec in config["config"]["layers"]:
                kind, c = spec["class_name"], spec["config"]
                if kind == "InputLayer":
                    input_shape = tuple(c["batch_input_shape"][1:])  # type: ignore[assignment]
                    continue
                if input_shape is None and c.get("batch_input_shape"):
                    input_shape = tuple(c["batch_input_shape"][1:])  # type: ignore[assignment]

                grp = mw[c["name"]]
                names = [_attr_text(n) for n in grp.attrs["weight_names"]]
                weights = [np.asarray(grp[n], dtype=np.float32) for n in names]

                if kind == "LSTM":
                    for flag in ("go_backwards", "stateful", "unroll"):
                        if c.get(flag):
                            raise UnsupportedModelError(f"LSTM {c['name']}: {flag}=True is not supported.")
                    if c.get("dropout") or c.get("recurrent_dropout"):
                        pass  # dropout is inactive at inference time
                    if c.get("recurrent_activation") != "sigmoid":
                        raise UnsupportedModelError(f"LSTM {c['name']}: recurrent_activation must be sigmoid.")
                    if c.get("activation") not in ("relu", "tanh"):
                        raise UnsupportedModelError(f"LSTM {c['name']}: unsupported activation {c.get('activation')!r}.")
                    if not c.get("use_bias", True) or len(weights) != 3:
                        raise UnsupportedModelError(f"LSTM {c['name']}: expected kernel, recurrent_kernel and bias.")
                    kernel, rec, bias = weights  # Keras order
                    units = int(c["units"])
                    if kernel.shape[1] != 4 * units or rec.shape != (units, 4 * units) or bias.shape != (4 * units,):
                        raise UnsupportedModelError(f"LSTM {c['name']}: weight shapes do not match units={units}.")
                    layers.append(
                        _Lstm(c["name"], kernel, rec, bias, units, _ACTIVATIONS[c["activation"]], bool(c["return_sequences"]))
                    )
                elif kind == "Dense":
                    act = c.get("activation", "linear")
                    if act not in _ACTIVATIONS:
                        raise UnsupportedModelError(f"Dense {c['name']}: unsupported activation {act!r}.")
                    if not c.get("use_bias", True) or len(weights) != 2:
                        raise UnsupportedModelError(f"Dense {c['name']}: expected kernel and bias.")
                    layers.append(_Dense(c["name"], weights[0], weights[1], _ACTIVATIONS[act]))
                else:
                    raise UnsupportedModelError(f"Unsupported layer type {kind!r} ({c.get('name')}).")

        if input_shape is None or len(input_shape) != 2:
            raise UnsupportedModelError("Could not determine a (timesteps, features) input shape.")
        last = layers[-1]
        if not isinstance(last, _Dense):
            raise UnsupportedModelError("Final layer must be Dense.")
        return cls(layers, input_shape, int(last.kernel.shape[1]), path)

    # --------------------------------------------------------------- predict
    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (sequence_length, feature_size) -> class probabilities (n_classes,)."""
        x = np.asarray(sequence, dtype=np.float32)
        if x.shape != (self.sequence_length, self.feature_size):
            raise ValueError(
                f"Expected input of shape {(self.sequence_length, self.feature_size)}, got {x.shape}."
            )
        if not np.isfinite(x).all():
            raise ValueError("Input contains NaN/Inf.")
        for layer in self._layers:
            x = layer.forward(x)
        return x
