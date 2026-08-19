"""Shared tetrahedral-mesh topology helpers.

Both the PDE assembly and the hierarchy export need the boundary of a tet mesh,
which comes down to finding the triangular faces that occur exactly once across
all tetrahedra.

The obvious spelling of that query, ``np.unique(faces, axis=0)``, views each row
as a structured dtype and compares rows field by field.  For an array with named
fields numpy takes its ``VOID_compare`` path, which performs a Python dictionary
lookup per field per comparison; on the multi-million-face meshes shipped with
this demo that single call dominates the entire run and looks indistinguishable
from a hang.

Packing each sorted index triple into one int64 key turns the same question into
a plain one-dimensional integer sort, which is several times faster at a few
million faces and does not degrade as the mesh grows.  When the index range is
too large for three indices to share an int64 the helpers fall back to
``np.unique(..., axis=0)``, which is still far better than the named-field path.
"""

from __future__ import annotations

import numpy as np

_INT64_MAX = int(np.iinfo(np.int64).max)


def _row_keys(rows: np.ndarray):
    """Pack each row of non-negative indices into one int64 key.

    Returns None when the values cannot be packed without overflow, which tells
    the caller to use the slower but unconditionally safe fallback.
    """
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    base = int(rows.max()) + 1
    if base <= 0 or base ** rows.shape[1] > _INT64_MAX:
        return None
    keys = rows[:, 0].astype(np.int64)
    for col in range(1, rows.shape[1]):
        keys *= base
        keys += rows[:, col]
    return keys


def row_multiplicity(rows: np.ndarray) -> np.ndarray:
    """For every row, how many rows of `rows` are identical to it.

    Parameters
    ----------
    rows : (N, K) integer array whose rows are already in canonical (sorted)
           order, so that two rows are equal exactly when they describe the
           same face.

    Returns
    -------
    (N,) int array of occurrence counts, aligned with `rows`.
    """
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.intp)
    keys = _row_keys(rows)
    if keys is None:
        _, inverse, counts = np.unique(rows, axis=0, return_inverse=True,
                                       return_counts=True)
    else:
        _, inverse, counts = np.unique(keys, return_inverse=True,
                                       return_counts=True)
    return counts[np.asarray(inverse).ravel()]


def first_occurrence_indices(rows: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each distinct row, for deduplication."""
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.intp)
    keys = _row_keys(rows)
    if keys is None:
        _, index = np.unique(rows, axis=0, return_index=True)
    else:
        _, index = np.unique(keys, return_index=True)
    return index
