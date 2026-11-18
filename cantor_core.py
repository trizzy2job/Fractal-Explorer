"""Mathematical and memory-bounded core for the complex self-affine explorer.

The finite stage consists of *words*, and duplicates are intentionally retained:
    C_k = {sum_{j=1}^k M^{-j} a_j : a_j in A}.
Here each point is stored as one complex128 row of length d.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import product
from math import isfinite

import numpy as np
from scipy.spatial import ConvexHull, QhullError

MAX_COMPLEX_DIM = 6
MAX_STAGE = 20  # UI defaults to 8; higher stages allowed with sampling.


@dataclass(frozen=True)
class FractalResult:
    points: np.ndarray
    words: int
    exact: bool
    stage: int
    alphabet_size: int
    expanding: bool
    eigenvalue_moduli: np.ndarray
    condition_number: float


@dataclass(frozen=True)
class CellEnvelope:
    center: np.ndarray  # real 2d-vector
    radii: np.ndarray   # half side lengths for each real dimension
    rigorous: bool
    description: str


@dataclass(frozen=True)
class CellShape:
    kind: str  # 'polygon', 'triangles', 'segment', 'point'
    vertices: np.ndarray  # 1xP, 2xP, or HxP
    faces: np.ndarray | None = None  # faces index vertices (for triangles)


def _parse_rows(text: str, columns: int, what: str) -> np.ndarray:
    rows = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            line = line[1:-1]
        try:
            values = ast.literal_eval('[' + line + ']')
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as exc:
            raise ValueError(f'{what} line {lineno}: invalid entry. Use Python complex notation (e.g. 1+2j).') from exc
        if len(values) != columns:
            raise ValueError(f'{what} line {lineno}: expected {columns} entries; found {len(values)}.')
        try:
            converted = np.asarray(values, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{what} line {lineno}: entries must be numbers.') from exc
        if not np.isfinite(converted).all():
            raise ValueError(f'{what} line {lineno}: entries must be finite.')
        rows.append(converted)
    if not rows:
        raise ValueError(f'{what}: enter at least one row.')
    return np.ascontiguousarray(np.stack(rows))


def parse_matrix(text: str, d: int) -> np.ndarray:
    if not 1 <= d <= MAX_COMPLEX_DIM:
        raise ValueError(f'd must be between 1 and {MAX_COMPLEX_DIM}.')
    matrix = _parse_rows(text, d, 'Matrix')
    if matrix.shape != (d, d):
        raise ValueError(f'Matrix: expected {d} rows, found {len(matrix)}.')
    return matrix


def parse_alphabet(text: str, d: int) -> np.ndarray:
    if not 1 <= d <= MAX_COMPLEX_DIM:
        raise ValueError(f'd must be between 1 and {MAX_COMPLEX_DIM}.')
    return _parse_rows(text, d, 'Alphabet')


def matrix_diagnostics(M: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    M = np.asarray(M, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError('M must be square.')
    if not np.isfinite(M).all():
        raise ValueError('M contains non-finite coefficients.')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            inv = np.linalg.inv(M)
            eig = np.linalg.eigvals(M)
            cond = float(np.linalg.cond(M))
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError('M must be numerically invertible.') from exc
    if not np.isfinite(inv).all() or not np.isfinite(eig).all():
        raise ValueError('M is too ill-conditioned for reliable computation.')
    return np.ascontiguousarray(inv), cond, np.abs(eig)


def _validate(M: np.ndarray, alphabet: np.ndarray, stage: int, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    M = np.asarray(M, dtype=np.complex128)
    alphabet = np.asarray(alphabet, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or not (1 <= M.shape[0] <= MAX_COMPLEX_DIM):
        raise ValueError('M must have square dimensions between 1 and 6.')
    d = M.shape[0]
    if alphabet.ndim != 2 or alphabet.shape[1] != d or alphabet.shape[0] < 1:
        raise ValueError('Alphabet must be a nonempty array of vectors in C^d.')
    if not np.isfinite(alphabet).all():
        raise ValueError('Alphabet must contain only finite numbers.')
    if not (isinstance(stage, int) and 0 <= stage <= MAX_STAGE):
        raise ValueError(f'Stage must be between 0 and {MAX_STAGE}.')
    if not (isinstance(limit, int) and 1 <= limit <= 2_000_000):
        raise ValueError('Point budget must be between 1 and 2,000,000.')
    inv, cond, eig = matrix_diagnostics(M)
    return M, alphabet, inv, cond, eig


def generate_points(M: np.ndarray, alphabet: np.ndarray, stage: int,
                    max_points: int = 150_000, seed: int = 20261106) -> FractalResult:
    """Compute C_k exactly if |A|**k <= max_points, else sample words uniformly.

    Exact construction uses O(|C_k| d) space plus one previous stage.
    Sampled construction uses exactly two (max_points,d) complex arrays,
    and blockwise allocations for alphabet indexing. All earlier stages are
    discarded. Samples are with replacement; distinctness is not assumed.
    """
    M, alphabet, B, cond, eig = _validate(M, alphabet, stage, max_points)
    d = M.shape[0]
    m = len(alphabet)
    words = m ** stage  # Python integer, never overflows.
    digit_B = np.ascontiguousarray(alphabet @ B.T)
    if words <= max_points:
        pts = np.zeros((1, d), dtype=np.complex128)
        for _ in range(stage):
            base = pts @ B.T
            count = len(pts)
            nxt = np.empty((count * m, d), dtype=np.complex128)
            for aidx, shift in enumerate(digit_B):
                np.add(base, shift, out=nxt[aidx * count:(aidx + 1) * count])
            pts = nxt
        exact = True
    else:
        # Draw random digit words, yielding the correct uniform distribution
        # over *sequences*, including potential collisions in their locations.
        rng = np.random.default_rng(seed)
        pts = np.zeros((max_points, d), dtype=np.complex128)
        work = np.empty_like(pts)
        chunk = max(512, min(16384, 131072 // d))
        for _ in range(stage):
            np.matmul(pts, B.T, out=work)
            for lo in range(0, max_points, chunk):
                hi = min(lo + chunk, max_points)
                indices = rng.integers(0, m, size=hi - lo, dtype=np.int32)
                np.add(work[lo:hi], digit_B[indices], out=work[lo:hi])
            pts, work = work, pts
        exact = False
    if not np.isfinite(pts).all():
        raise ValueError('Numeric overflow. Try a better-conditioned expanding matrix.')
    return FractalResult(pts, words, exact, stage, m,
                         bool(np.all(eig > 1.0 + 1e-10)), eig, cond)


def to_real(points: np.ndarray) -> np.ndarray:
    """Rows (z1,...,zd) become (Re z1,...,Re zd, Im z1,...,Im zd)."""
    z = np.asarray(points, dtype=np.complex128)
    return np.ascontiguousarray(np.concatenate((z.real, z.imag), axis=-1))


def real_matrix(B: np.ndarray) -> np.ndarray:
    """Realification [[Re B,-Im B],[Im B,Re B]]."""
    B = np.asarray(B, dtype=np.complex128)
    return np.block([[B.real, -B.imag], [B.imag, B.real]])


def compose_rotations(real_dim: int, rotations: list[tuple[int, int, float]]) -> np.ndarray:
    """Successive rotations (i,j,angle degrees), acting on column vectors."""
    R = np.eye(real_dim)
    for i, j, angle in rotations:
        if not (0 <= i < real_dim and 0 <= j < real_dim and i != j):
            raise ValueError('Rotation axes must be distinct and in range.')
        a = np.deg2rad(float(angle))
        co, si = np.cos(a), np.sin(a)
        # Rotate rows i and j of accumulated R, avoiding a full matrix product.
        ri, rj = R[i].copy(), R[j].copy()
        R[i] = co * ri - si * rj
        R[j] = si * ri + co * rj
    return R


def projection_matrix(real_dim: int, axes: tuple[int, ...],
                      rotations: list[tuple[int, int, float]]) -> np.ndarray:
    """Map R^(2d) to two or three visible coordinates; -1 means zero axis."""
    if len(axes) not in (2, 3):
        raise ValueError('Select 2 or 3 projection axes.')
    R = compose_rotations(real_dim, rotations)
    P = np.zeros((len(axes), real_dim))
    for k, axis in enumerate(axes):
        if axis != -1:
            if not 0 <= axis < real_dim:
                raise ValueError('Projection axis is outside the ambient dimension.')
            P[k] = R[axis]
    return P


def project(points: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project complex point array with no persistent realified copy."""
    d = points.shape[1]
    if P.shape[1] != 2*d:
        raise ValueError('Projection matrix dimension mismatch.')
    # BLAS on contiguous real/imag matrices; avoids concatenating a large
    # N x 2d realified array for every slider movement.
    return points.real @ P[:, :d].T + points.imag @ P[:, d:].T


def invariant_envelope(M: np.ndarray, alphabet: np.ndarray) -> CellEnvelope:
    """Find a real-axis-aligned cube/box Q containing the full attractor K.

    With B=M^-1 and f_a(x)=B(x+a), first solve the interval fixed point
      r >= |T|r + max_a |T c + T a-c|
    for a center c fixed under the mean digit. This is rigorous when
    rho(|T|)<1. If rho(|T|)>=1 but rho(B)<1, a norm-power bound gives a
    (looser) cube. For nonexpanding M, returns a non-covering reference cube.
    """
    M = np.asarray(M, dtype=np.complex128)
    alphabet = np.asarray(alphabet, dtype=np.complex128)
    B, _, eig = matrix_diagnostics(M)
    T = real_matrix(B)
    n = len(T)
    b_digits = to_real(alphabet @ B.T)
    if not np.all(eig > 1 + 1e-10):
        # Well-defined finite-stage geometry; no attractor containment claim.
        return CellEnvelope(np.zeros(n), np.ones(n)*0.5, False,
                            'Reference cells only: M is not strictly expanding.')
    try:
        c = np.linalg.solve(np.eye(n) - T, b_digits.mean(axis=0))
        offsets = c @ T.T + b_digits - c
        v = np.max(np.abs(offsets), axis=0)
        S = np.abs(T)
        if np.max(np.abs(np.linalg.eigvals(S))) < 1 - 1e-10:
            r = np.linalg.solve(np.eye(n) - S, v)
            if np.isfinite(r).all() and np.all(r >= -1e-8):
                # Small inflation guards numerical roundoff.
                r = np.maximum(r, 0) * (1 + 1e-10)
                return CellEnvelope(c, r, True, 'Invariant coordinate box (rigorous attractor cover).')
    except np.linalg.LinAlgError:
        pass

    # Bound sum_{j>=1} ||B^j||_2 via some L with ||B^L||<1:
    # sum_{j>=1} ||B^j|| <= sum_{r=1}^L ||B^r|| / (1-||B^L||).
    accum = np.eye(len(B), dtype=np.complex128)
    norm_sum = 0.0
    for L in range(1, 513):
        accum = accum @ B
        q = float(np.linalg.norm(accum, ord=2))
        norm_sum += q
        if q < 0.8:
            radius = (norm_sum / (1-q))*float(np.max(np.linalg.norm(alphabet, axis=1)))
            if np.isfinite(radius):
                return CellEnvelope(np.zeros(n), np.full(n, radius*(1+1e-10)), True,
                                    f'Conservative norm-bound cube (L={L}; rigorous cover).')
            break
    return CellEnvelope(np.zeros(n), np.ones(n)*0.5, False,
                        'Reference cells only: stable numerical attractor bound unavailable.')


def projected_cell_geometry(M: np.ndarray, stage: int, P: np.ndarray,
                             envelope: CellEnvelope) -> tuple[np.ndarray, CellShape]:
    """Return projected cell-center shift and a common projected zonotope.

    Each k-cylinder is t_word + M^-k Q (Q is the invariant envelope).
    Its projection is the translation P*t_word plus `center_shift`
    plus a common polytope given by `shape`.
    """
    if stage < 0:
        raise ValueError('Stage must be nonnegative.')
    B, _, _ = matrix_diagnostics(M)
    Tpow = np.linalg.matrix_power(real_matrix(B), stage)
    shift = P @ (Tpow @ envelope.center)
    generators = (P @ Tpow) * envelope.radii[None, :]
    p, n = generators.shape
    if p not in (2, 3):
        raise ValueError('Visualization requires 2 or 3 projected dimensions.')
    if not np.isfinite(generators).all():
        raise ValueError('Cell geometry overflowed.')
    # All 2^12 vertices cost < 0.4 MiB, regardless of number of cells.
    # Ignore exactly-zero generators to reduce corner count.
    lengths = np.linalg.norm(generators, axis=0)
    active = lengths > max(1e-14, np.max(lengths, initial=0)*1e-13)
    G = generators[:, active]
    if G.shape[1] == 0:
        return shift, CellShape('point', np.zeros((1,p)))
    sgn = np.asarray(list(product((-1.0, 1.0), repeat=G.shape[1])), dtype=np.float64)
    vertices = sgn @ G.T
    u, s, _ = np.linalg.svd(G, full_matrices=True)
    rank = int(np.sum(s > max(1e-13, s[0]*1e-10)))
    if rank == 0:
        return shift, CellShape('point', np.zeros((1,p)))
    if rank == 1:
        direction = u[:, 0]
        extent = np.sum(np.abs(direction @ G))
        return shift, CellShape('segment', np.stack((-extent * direction, extent * direction)))
    if rank == 2 and p == 3:
        basis = u[:, :2]
        local = vertices @ basis
        hull = ConvexHull(local)
        return shift, CellShape('polygon', vertices[hull.vertices])
    try:
        hull = ConvexHull(vertices)
    except QhullError as exc:
        raise ValueError('Unable to construct projected cell hull: numerical degeneracy.') from exc
    if p == 2:
        return shift, CellShape('polygon', vertices[hull.vertices])
    # Coalesce rounded duplicate triangles to avoid redundant geometry.
    return shift, CellShape('triangles', vertices, hull.simplices)
