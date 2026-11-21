"""SM-CTRN: Sparse Modular Continuous-Time Recurrent Network.

This is the non-Transformer cognitive core required by MASTER SPEC 5.1. It is a
plain continuous-time recurrent network with block-sparse recurrent weights and
explicit inter-module coupling plus a shared concept state. **It is not claimed
to be a novel invention**; sparsity, CT-RNN dynamics and Hebbian-style learning
all pre-exist. What the project studies is the interaction of this core with
memory, affect modulation, concept formation and world/language co-evolution.

Dynamics per module ``i`` (Euler integration)::

    h_i(t+dt) = h_i(t) + (dt/tau_i) * (-h_i(t) + phi(W_ii h_i + sum_j C_ij h_j
                                                    + P_i c + U_i x_i + b_i))

and for the shared concept state ``c``::

    c(t+dt)   = c(t) + (dt/tau_c) * (-c(t) + phi(Q c_inputs))

Representation note: masks are stored as boolean arrays and weights as dense
``float64`` matrices of shape (total_units, total_units). At the configured
scale (a few hundred units) a dense BLAS matmul is far faster than any Python
sparse product, while the *connectivity* is still genuinely sparse and is
reported by :meth:`SparseModularCTRN.density_report`. ``to_csr``/``from_csr``
provide an explicit sparse export for audit and future larger scales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from wphr.core.registry import MODULE_ORDER
from wphr.util.numerics import sanitize_array, spectral_radius
from wphr.version import SCHEMA_VERSION


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    units: int
    tau: float
    input_dim: int


@dataclass
class CoreMetrics:
    h_norm: float
    c_norm: float
    saturation: float
    faults: int
    step: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "h_norm": round(self.h_norm, 6),
            "c_norm": round(self.c_norm, 6),
            "saturation": round(self.saturation, 6),
            "faults": self.faults,
            "step": self.step,
        }


class SparseModularCTRN:
    """Sparse modular continuous-time recurrent network with a shared concept state."""

    def __init__(
        self,
        core_cfg: dict[str, Any],
        seed: int,
        *,
        dt: float = 0.1,
        finite_policy: str = "sanitize",
        module_order: tuple[str, ...] | None = None,
    ) -> None:
        if finite_policy not in ("raise", "sanitize"):
            raise ValueError("finite_policy must be 'raise' or 'sanitize'")
        self.cfg = dict(core_cfg)
        self.dt = float(dt)
        self.finite_policy = finite_policy
        self.seed = int(seed)
        self.sparsity = float(core_cfg["sparsity"])
        self.weight_scale = float(core_cfg["weight_scale"])
        self.weight_bound = float(core_cfg["weight_bound"])
        self.state_bound = float(core_cfg["state_bound"])
        self.activation = str(core_cfg.get("activation", "tanh"))
        self.concept_dim = int(core_cfg["concept_dim"])
        self.concept_tau = float(core_cfg["concept_tau"])
        self.spectral_target = float(core_cfg.get("spectral_target", 0.95))

        order = tuple(module_order or MODULE_ORDER)
        modules = core_cfg["modules"]
        rank = {name: i for i, name in enumerate(order)}
        names = sorted(modules, key=lambda n: (rank.get(n, len(rank)), n))
        self.specs: list[ModuleSpec] = [
            ModuleSpec(
                name=name,
                units=int(modules[name]["units"]),
                tau=float(modules[name]["tau"]),
                input_dim=int(modules[name]["input_dim"]),
            )
            for name in names
        ]
        self.module_names = tuple(spec.name for spec in self.specs)
        self.total_units = sum(spec.units for spec in self.specs)
        self.offsets: dict[str, tuple[int, int]] = {}
        self.input_offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        input_cursor = 0
        for spec in self.specs:
            self.offsets[spec.name] = (cursor, cursor + spec.units)
            cursor += spec.units
            self.input_offsets[spec.name] = (input_cursor, input_cursor + spec.input_dim)
            input_cursor += spec.input_dim
        self.total_inputs = input_cursor

        self.tau = np.concatenate([np.full(spec.units, spec.tau) for spec in self.specs])
        self._rng = np.random.default_rng(self.seed)
        self._init_parameters()

        self.h = np.zeros(self.total_units, dtype=np.float64)
        self.c = np.zeros(self.concept_dim, dtype=np.float64)
        self.step_index = 0
        self.faults = 0
        self.rho = self._normalize_spectrum()

    def _normalize_spectrum(self) -> float:
        """Scale the coupling matrix so the linearised dynamics stay contractive.

        Without this, random sparse weights can produce a spectral radius > 1 and
        a saturated (or divergent) trajectory. Scaling preserves the sparse
        structure and relative weights; it is a stability measure, not a claim
        about biological plausibility.
        """
        rho = self.recompute_spectral()
        if rho > self.spectral_target > 0:
            scale = self.spectral_target / rho
            self.W *= scale
            self.P *= scale
            self.Q *= scale
            rho = self.recompute_spectral()
        return rho

    # ------------------------------------------------------------- parameters
    def _init_parameters(self) -> None:
        rng = self._rng
        n = self.total_units
        mask = np.zeros((n, n), dtype=bool)
        for i, mi in enumerate(self.specs):
            si, ei = self.offsets[mi.name]
            for j, mj in enumerate(self.specs):
                sj, ej = self.offsets[mj.name]
                block = rng.random((mi.units, mj.units)) < self.sparsity
                if i == j:
                    block = self._ensure_rows(block, rng)
                mask[si:ei, sj:ej] = block
        self.mask = mask

        w = np.where(mask, rng.normal(0.0, 1.0, size=(n, n)), 0.0)
        self.W = self._row_scale(w, mask)

        p_mask = self._ensure_rows(rng.random((n, self.concept_dim)) < self.sparsity, rng)
        self.P_mask = p_mask
        self.P = self._row_scale(np.where(p_mask, rng.normal(0.0, 1.0, (n, self.concept_dim)), 0.0), p_mask)

        q_mask = self._ensure_rows(rng.random((self.concept_dim, n)) < self.sparsity, rng)
        self.Q_mask = q_mask
        self.Q = self._row_scale(np.where(q_mask, rng.normal(0.0, 1.0, (self.concept_dim, n)), 0.0), q_mask)

        u_mask = np.zeros((n, self.total_inputs), dtype=bool)
        for spec in self.specs:
            si, ei = self.offsets[spec.name]
            ii, ij = self.input_offsets[spec.name]
            u_mask[si:ei, ii:ij] = rng.random((spec.units, spec.input_dim)) < 0.5
        self.U_mask = u_mask
        self.U = self._row_scale(np.where(u_mask, rng.normal(0.0, 1.0, (n, self.total_inputs)), 0.0), u_mask)

        self.b = np.zeros(n, dtype=np.float64)
        self.bc = np.zeros(self.concept_dim, dtype=np.float64)

    def _row_scale(self, matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
        fan_in = mask.sum(axis=1).astype(np.float64)
        fan_in[fan_in == 0] = 1.0
        scale = self.weight_scale / np.sqrt(fan_in)
        return matrix * scale[:, None]

    @staticmethod
    def _ensure_rows(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Guarantee at least one incoming edge per row so no unit is dead."""
        empty = ~mask.any(axis=1)
        if empty.any():
            cols = rng.integers(0, mask.shape[1], size=int(empty.sum()))
            mask[np.nonzero(empty)[0], cols] = True
        return mask

    # --------------------------------------------------------------- dynamics
    def _activation(self, pre: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(pre)
        return np.clip(pre, -1.0, 1.0)

    def build_input(self, values: dict[str, np.ndarray] | None) -> np.ndarray:
        """Assemble the flat input vector from per-module input vectors."""
        flat = np.zeros(self.total_inputs, dtype=np.float64)
        if not values:
            return flat
        for name, vec in values.items():
            if name not in self.input_offsets:
                raise KeyError(f"unknown module input: {name}")
            start, end = self.input_offsets[name]
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            expected = end - start
            if arr.size != expected:
                raise ValueError(
                    f"input for {name} must have {expected} values, got {arr.size}"
                )
            flat[start:end] = arr
        return flat

    def step(self, x: np.ndarray | None = None, dt: float | None = None) -> CoreMetrics:
        dt_value = self.dt if dt is None else float(dt)
        if dt_value <= 0:
            raise ValueError("dt must be positive")
        flat = np.zeros(self.total_inputs, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
        if flat.ndim != 1 or flat.size != self.total_inputs:
            raise ValueError(
                f"input vector must be shape ({self.total_inputs},), got {flat.shape}"
            )

        pre = self.W @ self.h + self.P @ self.c + self.U @ flat + self.b
        act = self._activation(pre)
        ratio = dt_value / self.tau
        h_new = self.h + ratio * (-self.h + act)
        h_new, faults_h = self._guard(h_new, "hidden")

        pre_c = self.Q @ self.h + self.bc
        c_new = self.c + (dt_value / self.concept_tau) * (-self.c + np.tanh(pre_c))
        c_new, faults_c = self._guard(c_new, "concept")

        self.h = h_new
        self.c = c_new
        self.step_index += 1
        self.faults += faults_h + faults_c
        saturation = float(np.mean(np.abs(act) > 0.95))
        return CoreMetrics(
            h_norm=float(np.linalg.norm(self.h)),
            c_norm=float(np.linalg.norm(self.c)),
            saturation=saturation,
            faults=self.faults,
            step=self.step_index,
        )

    def _guard(self, array: np.ndarray, where: str) -> tuple[np.ndarray, int]:
        if self.finite_policy == "raise":
            if not np.isfinite(array).all():
                raise FloatingPointError(f"non-finite values in {where} state")
            np.clip(array, -self.state_bound, self.state_bound, out=array)
            return array, 0
        return sanitize_array(array, self.state_bound, where)

    # ------------------------------------------------------------ diagnostics
    def messages(self, h: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Explicit module-to-module messages for the current hidden state."""
        state = self.h if h is None else np.asarray(h, dtype=np.float64)
        out: dict[str, np.ndarray] = {}
        for src in self.specs:
            s_src, e_src = self.offsets[src.name]
            for dst in self.specs:
                if dst.name == src.name:
                    continue
                s_dst, _ = self.offsets[dst.name]
                block = self.W[s_dst : s_dst + dst.units, s_src:e_src]
                out[f"{src.name}->{dst.name}"] = block @ state[s_src:e_src]
        return out

    def density_report(self) -> dict[str, Any]:
        def density(mask: np.ndarray) -> float:
            mask = np.asarray(mask, dtype=bool)
            return float(mask.sum() / mask.size) if mask.size else 0.0

        return {
            "total_units": int(self.total_units),
            "concept_dim": int(self.concept_dim),
            "configured_sparsity": self.sparsity,
            "recurrent_density": density(self.mask),
            "recurrent_edges": int(self.mask.sum()),
            "concept_in_density": density(self.P_mask),
            "concept_out_density": density(self.Q_mask),
            "input_density": density(self.U_mask),
            "parameters": int(
                self.W.size + self.P.size + self.Q.size + self.U.size + self.b.size + self.bc.size
            ),
            "nonzero_parameters": int(
                np.count_nonzero(self.W)
                + np.count_nonzero(self.P)
                + np.count_nonzero(self.Q)
                + np.count_nonzero(self.U)
            ),
        }

    def coupling_matrix(self) -> np.ndarray:
        """Linearised state matrix M = [[W, P], [Q, 0]] over [h; c]."""
        n, m = self.total_units, self.concept_dim
        mat = np.zeros((n + m, n + m), dtype=np.float64)
        mat[:n, :n] = self.W
        mat[:n, n:] = self.P
        mat[n:, :n] = self.Q
        return mat

    def recompute_spectral(self) -> float:
        return spectral_radius(self.coupling_matrix())

    def sanity(self) -> dict[str, Any]:
        finite = bool(np.isfinite(self.h).all() and np.isfinite(self.c).all())
        return {
            "finite": finite,
            "faults": self.faults,
            "spectral_radius": round(self.rho, 6),
            "spectral_target": self.spectral_target,
            "h_norm": float(np.linalg.norm(self.h)),
            "c_norm": float(np.linalg.norm(self.c)),
        }

    # ------------------------------------------------------------ checkpoint
    def state_arrays(self) -> dict[str, np.ndarray]:
        rng_state = json.dumps(self._rng.bit_generator.state, sort_keys=True)
        return {
            "h": self.h.copy(),
            "c": self.c.copy(),
            "W": self.W.copy(),
            "P": self.P.copy(),
            "Q": self.Q.copy(),
            "U": self.U.copy(),
            "b": self.b.copy(),
            "bc": self.bc.copy(),
            "mask": self.mask.copy(),
            "P_mask": self.P_mask.copy(),
            "Q_mask": self.Q_mask.copy(),
            "U_mask": self.U_mask.copy(),
            "tau": self.tau.copy(),
            "_meta": np.array(
                json.dumps(
                    {
                        "step_index": self.step_index,
                        "faults": self.faults,
                        "rho": self.rho,
                        "seed": self.seed,
                        "schema_version": SCHEMA_VERSION,
                        "module_names": list(self.module_names),
                        "rng_state": rng_state,
                    },
                    sort_keys=True,
                )
            ),
        }

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        meta = json.loads(str(arrays["_meta"].item()))
        if list(meta.get("module_names", [])) != list(self.module_names):
            raise ValueError("checkpoint module layout does not match the configured core")
        for name in ("W", "P", "Q", "U", "b", "bc", "mask", "P_mask", "Q_mask", "U_mask", "tau"):
            arr = np.asarray(arrays[name], dtype=bool if name.endswith("mask") else np.float64)
            current = getattr(self, name)
            if arr.shape != current.shape:
                raise ValueError(f"checkpoint array {name} has shape {arr.shape}, expected {current.shape}")
            setattr(self, name, arr.astype(current.dtype, copy=True))
        self.h = np.asarray(arrays["h"], dtype=np.float64).copy()
        self.c = np.asarray(arrays["c"], dtype=np.float64).copy()
        self.step_index = int(meta.get("step_index", 0))
        self.faults = int(meta.get("faults", 0))
        self.rho = float(meta.get("rho", self.recompute_spectral()))
        rng_state = meta.get("rng_state")
        if rng_state:
            self._rng.bit_generator.state = json.loads(rng_state)

    # ---------------------------------------------------------------- sparse
    def to_csr(self) -> dict[str, np.ndarray]:
        """Explicit sparse export (row indices, sorted column indices, values)."""
        rows, cols = np.nonzero(self.W)
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        return {
            "shape": np.array(self.W.shape),
            "rows": rows.astype(np.int32),
            "cols": cols.astype(np.int32),
            "values": self.W[rows, cols].copy(),
        }

    @classmethod
    def from_csr(cls, csr: dict[str, np.ndarray]) -> np.ndarray:
        shape = tuple(int(x) for x in np.asarray(csr["shape"]).tolist())
        dense = np.zeros(shape, dtype=np.float64)
        dense[csr["rows"], csr["cols"]] = csr["values"]
        return dense
