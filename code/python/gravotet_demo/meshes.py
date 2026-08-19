"""Tet-mesh inputs for the GravoTet supplementary demos.

Two entry points:

* :func:`make_cube_mesh` - regular cube, built directly by the C++
  extension via ``MultigridSolver.generate_cube_mesh(resolution)``.
* :func:`make_mesh_from_ply` - load any tet mesh (binary or ASCII PLY)
  from disk into a ``gravotet.TetrahedralMesh``.

The non-cube demos (spot, sphere, torus) load TetGen-quality tet meshes that
exceed GitHub's 100 MB per-file limit.  They are committed as sub-100 MB parts
under ``GravoTet/data_chunks/`` and concatenated back into ``GravoTet/data/``
on first use (see :mod:`gravotet_demo.data_assembly`).  The supplement itself
does not generate these meshes; it loads them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Binary PLY I/O
# ---------------------------------------------------------------------------

def _save_tet_ply(path: Path, vertices: np.ndarray, tetrahedra: np.ndarray) -> None:
    """Write a tetrahedral mesh as binary little-endian PLY.

    Tets are stored as faces with a 4-vertex list, which is the convention
    that loaders such as Blender and the main project's ``create_tet_meshes``
    pipeline use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    verts32 = np.ascontiguousarray(vertices, dtype="<f4")
    tets32 = np.ascontiguousarray(tetrahedra, dtype="<i4")

    face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", 4)])
    face_records = np.empty(len(tets32), dtype=face_dtype)
    face_records["count"] = 4
    face_records["idx"] = tets32

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(verts32)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(tets32)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts32.tobytes())
        f.write(face_records.tobytes())


def _parse_ply_header(f, path: Path) -> Tuple[int, int, bool]:
    """Read and validate the PLY header, leaving `f` positioned at the body.

    Returns ``(num_vertices, num_faces, is_binary)``.  The reader below assumes
    exactly three float vertex properties (x, y, z) and a face property list of
    4-index entries, which is the layout written by :func:`_save_tet_ply` and by
    the project's TetGen conversion.  Anything else is rejected here with a
    specific message instead of being silently misread as garbage geometry.
    """
    num_vertices = num_faces = 0
    is_binary = False
    is_little_endian = True
    element = None
    vertex_properties: list = []
    face_properties: list = []
    saw_end_header = False

    first = f.readline()
    if first.strip() not in (b"ply", b"ply\r"):
        raise ValueError(f"{path} is not a PLY file (bad magic line).")

    while True:
        raw = f.readline()
        if not raw:
            raise ValueError(f"Unexpected end of file while reading the header of {path}.")
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        keyword = parts[0]

        if keyword == "format":
            is_binary = "binary" in line
            is_little_endian = "big_endian" not in line
        elif keyword == "element":
            element = parts[1]
            if element == "vertex":
                num_vertices = int(parts[2])
            elif element == "face":
                num_faces = int(parts[2])
        elif keyword == "property":
            if element == "vertex":
                vertex_properties.append(parts[-1])
            elif element == "face":
                face_properties.append(parts)
        elif keyword == "end_header":
            saw_end_header = True
            break

    if not saw_end_header:
        raise ValueError(f"Missing end_header in {path}.")
    if is_binary and not is_little_endian:
        raise ValueError(
            f"{path} is a big-endian binary PLY, which this reader does not "
            f"support.  Convert it to binary_little_endian or ASCII."
        )
    if vertex_properties[:3] != ["x", "y", "z"] or len(vertex_properties) != 3:
        raise ValueError(
            f"{path} has vertex properties {vertex_properties}; this reader "
            f"expects exactly three float properties x, y, z.  A PLY carrying "
            f"normals or colours has to be converted first."
        )
    if is_binary and face_properties:
        spec = face_properties[0]
        # e.g. ['property', 'list', 'uchar', 'int', 'vertex_indices']
        if len(spec) >= 4 and (spec[2] not in ("uchar", "uint8") or
                               spec[3] not in ("int", "int32", "uint", "uint32")):
            raise ValueError(
                f"{path} declares the face list as '{' '.join(spec[1:4])}'; the "
                f"binary reader expects 'list uchar int'."
            )
    return num_vertices, num_faces, is_binary


def _stream_ascii_blocks(f, block_size: int = 8 << 20):
    """Yield decoded text blocks that always end on a line boundary.

    Reading the body in bounded blocks keeps peak memory flat: the alternative
    (``f.read().splitlines()``) materialises one Python string per line, which
    for the 4.9 M-line spot mesh alone costs hundreds of megabytes.
    """
    remainder = b""
    while True:
        block = f.read(block_size)
        if not block:
            tail = remainder.strip()
            if tail:
                yield tail.decode("ascii", errors="replace")
            return
        block = remainder + block
        cut = block.rfind(b"\n")
        if cut == -1:
            remainder = block
            continue
        remainder = block[cut + 1:]
        yield block[:cut].decode("ascii", errors="replace")


def _numbers_from_text(text: str) -> np.ndarray:
    """Parse whitespace-separated numbers into a float64 array.

    ``np.fromstring`` in text mode parses straight from the C buffer and is
    about twice as fast as ``np.array(text.split(), ...)``, which has to build
    one Python string per token first.  The slower form is kept as a fallback in
    case a future numpy drops the text mode.

    Vertex coordinates and connectivity are both parsed as float64.  Vertex
    indices are far below 2**53, so the round-trip through a double is exact.
    """
    try:
        return np.fromstring(text, dtype=np.float64, sep=" ")
    except (ValueError, DeprecationWarning):
        return np.array(text.split(), dtype=np.float64)


def _load_ascii_body(f, path: Path, num_vertices: int,
                     num_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the ASCII body of a tetrahedral PLY.

    The body is one flat token stream: ``3 * num_vertices`` coordinate tokens
    followed by ``5 * num_faces`` face tokens (a leading vertex count of 4 plus
    the four indices).  Tokens are converted in blocks with a single numpy call
    each, which is roughly an order of magnitude faster than a per-line Python
    loop and keeps memory bounded.
    """
    n_vert_tokens = num_vertices * 3
    n_face_tokens = num_faces * 5
    total = n_vert_tokens + n_face_tokens

    verts_flat = np.empty(n_vert_tokens, dtype=np.float64)
    faces_flat = np.empty(n_face_tokens, dtype=np.int64)

    seen = 0
    for text in _stream_ascii_blocks(f):
        if seen >= total:
            break
        values = _numbers_from_text(text)
        if seen + values.size > total:
            values = values[: total - seen]
        start, end = seen, seen + values.size

        hi = min(end, n_vert_tokens)
        if hi > start:
            verts_flat[start:hi] = values[: hi - start]
        lo = max(start, n_vert_tokens)
        if end > lo:
            faces_flat[lo - n_vert_tokens: end - n_vert_tokens] = values[lo - start:]
        seen = end

    if seen < total:
        raise ValueError(
            f"{path} is truncated: the header declares {num_vertices:,} vertices "
            f"and {num_faces:,} tetrahedra ({total:,} values) but only {seen:,} "
            f"were found.  Re-assemble it with 'python scripts/fetch_data.py "
            f"--force'."
        )

    faces = faces_flat.reshape(num_faces, 5)
    if num_faces and not np.all(faces[:, 0] == 4):
        raise ValueError(
            f"{path} contains faces that are not 4-vertex tetrahedra; this "
            f"reader only handles tetrahedral PLY files."
        )
    return (verts_flat.reshape(num_vertices, 3),
            np.ascontiguousarray(faces[:, 1:5], dtype=np.int32))


def _load_tet_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a tetrahedral PLY (binary little-endian or ASCII).

    Raises :class:`ValueError` with an actionable message when the file is
    truncated, corrupt, or in a layout this reader does not handle, rather than
    letting a downstream reshape fail with an opaque numpy error.
    """
    with open(path, "rb") as f:
        num_vertices, num_faces, is_binary = _parse_ply_header(f, path)

        if is_binary:
            vert_bytes = num_vertices * 12
            raw_verts = f.read(vert_bytes)
            if len(raw_verts) != vert_bytes:
                raise ValueError(
                    f"{path} is truncated: expected {vert_bytes:,} bytes of "
                    f"vertex data, found {len(raw_verts):,}.  Re-assemble it "
                    f"with 'python scripts/fetch_data.py --force'."
                )
            verts32 = np.frombuffer(raw_verts, dtype="<f4").reshape(num_vertices, 3)

            face_dtype = np.dtype([("count", "u1"), ("idx", "<i4", 4)])
            face_bytes = num_faces * face_dtype.itemsize
            raw_faces = f.read(face_bytes)
            if len(raw_faces) != face_bytes:
                raise ValueError(
                    f"{path} is truncated: expected {face_bytes:,} bytes of "
                    f"tetrahedron data, found {len(raw_faces):,}.  Re-assemble "
                    f"it with 'python scripts/fetch_data.py --force'."
                )
            faces = np.frombuffer(raw_faces, dtype=face_dtype)
            if num_faces and not np.all(faces["count"] == 4):
                raise ValueError(
                    f"{path} contains faces that are not 4-vertex tetrahedra; "
                    f"this reader only handles tetrahedral PLY files."
                )
            return (verts32.astype(np.float64),
                    np.ascontiguousarray(faces["idx"], dtype=np.int32))

        return _load_ascii_body(f, path, num_vertices, num_faces)


# ---------------------------------------------------------------------------
# Public mesh entry points
# ---------------------------------------------------------------------------

def make_cube_mesh(gravotet: Any, resolution: int) -> Any:
    """Return a ``TetrahedralMesh`` for a regular cube at the given resolution.

    The vertex count is ``resolution**3``.  Resolution 47 yields 103,823
    vertices; resolution 100 yields one million.
    """
    return gravotet.MultigridSolver.generate_cube_mesh(resolution)


_MISSING_PLY_HINT = (
    "\n\nThe spot, sphere, and torus meshes are committed as sub-100 MB parts "
    "under data_chunks/ and are normally assembled automatically.  If the parts "
    "are also missing, rebuild them from source meshes with:\n"
    "    python scripts/split_meshes.py\n"
    "The cube demo (run_demo_cube_200k.py) needs no input file.\n"
)


def make_mesh_from_ply(gravotet: Any, ply_path: Path,
                       verbose: bool = True) -> Any:
    """Load a tetrahedral PLY and return a ``gravotet.TetrahedralMesh``.

    If the file is a bundled mesh that has not been assembled yet, its parts
    under ``data_chunks/`` are concatenated automatically.  Raises
    :class:`FileNotFoundError` with an actionable hint if the file (and its
    parts) cannot be found.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        from .data_assembly import ensure_mesh_for_path
        ensure_mesh_for_path(ply_path, verbose=verbose)
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Required input mesh not found: {ply_path}{_MISSING_PLY_HINT}"
        )

    if verbose:
        print(f"  Loading tet mesh: {ply_path}")
    vertices, tetrahedra = _load_tet_ply(ply_path)
    if verbose:
        print(f"  {len(vertices):,} vertices, {len(tetrahedra):,} tetrahedra")

    mesh = gravotet.TetrahedralMesh()
    mesh.vertices = vertices
    mesh.tetrahedra = tetrahedra
    return mesh
