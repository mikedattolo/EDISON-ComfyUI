"""
CAD / mesh quality-assurance gates.

Phase 3 goal: catch print-killing geometry before slicing. Implemented as
optional gates so callers can pick which checks to run:

* manifold check (closed surface, every edge shared by 2 faces)
* min wall-thickness check (heuristic: shortest edge length)
* bounding-box sanity (build-volume fit)
* triangle count + degenerate-face report

These gates intentionally avoid pulling in ``trimesh`` or ``open3d`` as
hard dependencies. We use them when available and fall back to a pure
STL (binary or ASCII) parser so the QA layer always works.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QAReport:
    path: str
    triangles: int
    bounding_box: Dict[str, List[float]]
    is_manifold: bool
    open_edges: int
    min_edge_length_mm: float
    max_edge_length_mm: float
    degenerate_faces: int
    fits_build_volume: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "triangles": self.triangles,
            "bounding_box": self.bounding_box,
            "is_manifold": self.is_manifold,
            "open_edges": self.open_edges,
            "min_edge_length_mm": self.min_edge_length_mm,
            "max_edge_length_mm": self.max_edge_length_mm,
            "degenerate_faces": self.degenerate_faces,
            "fits_build_volume": self.fits_build_volume,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "passed": self.passed,
        }


# ── STL parsing ───────────────────────────────────────────────────────

def _parse_stl(path: Path) -> List[Tuple[Tuple[float, float, float], ...]]:
    """Parse an STL file into a list of triangles (each is 3 vertex tuples).

    Supports both binary and ASCII STL.
    """
    data = path.read_bytes()
    if len(data) >= 84:
        # Detect ASCII by checking for "solid" header *and* failing to parse
        # binary triangle count cleanly.
        try:
            tri_count = struct.unpack_from("<I", data, 80)[0]
            expected = 84 + tri_count * 50
            if expected == len(data):
                return _parse_binary_stl(data, tri_count)
        except struct.error:
            pass
    return _parse_ascii_stl(data.decode("utf-8", errors="replace"))


def _parse_binary_stl(data: bytes, tri_count: int) -> List[Tuple[Tuple[float, float, float], ...]]:
    triangles: List[Tuple[Tuple[float, float, float], ...]] = []
    offset = 84
    for _ in range(tri_count):
        # 12 floats: normal(3) + v0(3) + v1(3) + v2(3), then 2 byte attr
        floats = struct.unpack_from("<12f", data, offset)
        v0 = (floats[3], floats[4], floats[5])
        v1 = (floats[6], floats[7], floats[8])
        v2 = (floats[9], floats[10], floats[11])
        triangles.append((v0, v1, v2))
        offset += 50
    return triangles


def _parse_ascii_stl(text: str) -> List[Tuple[Tuple[float, float, float], ...]]:
    triangles: List[Tuple[Tuple[float, float, float], ...]] = []
    current: List[Tuple[float, float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    current.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
            if len(current) == 3:
                triangles.append(tuple(current))
                current = []
        elif line.startswith("endfacet"):
            current = []
    return triangles


# ── Geometry helpers ─────────────────────────────────────────────────

def _edge_key(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    *,
    ndigits: int = 4,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return a canonical edge key with vertices rounded to ``ndigits``.

    Rounding is essential — STL stores vertices as float32, and shared
    edges between adjacent triangles can drift in the lowest bits during
    save/load. Without rounding, every shared edge looks unique and the
    manifold check produces false positives.
    """
    ra = (round(a[0], ndigits), round(a[1], ndigits), round(a[2], ndigits))
    rb = (round(b[0], ndigits), round(b[1], ndigits), round(b[2], ndigits))
    return (ra, rb) if ra <= rb else (rb, ra)


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _bounding_box(triangles: List[Tuple[Tuple[float, float, float], ...]]) -> Dict[str, List[float]]:
    if not triangles:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0]}
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for tri in triangles:
        for v in tri:
            xs.append(v[0]); ys.append(v[1]); zs.append(v[2])
    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
        "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
    }


# ── Public API ───────────────────────────────────────────────────────

def run_qa(
    mesh_path: str | Path,
    *,
    min_wall_thickness_mm: float = 0.4,
    build_volume_mm: Optional[Tuple[float, float, float]] = None,
) -> QAReport:
    """Run all QA gates against an STL file and return a structured report.

    ``min_wall_thickness_mm`` defaults to 0.4mm (typical FDM nozzle width).
    ``build_volume_mm`` should be ``(x, y, z)`` of the target printer; when
    provided we set ``fits_build_volume``.
    """
    path = Path(mesh_path)
    report = QAReport(
        path=str(path),
        triangles=0,
        bounding_box={"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0]},
        is_manifold=False,
        open_edges=0,
        min_edge_length_mm=0.0,
        max_edge_length_mm=0.0,
        degenerate_faces=0,
    )

    if not path.exists():
        report.errors.append(f"mesh not found: {path}")
        report.passed = False
        return report
    if path.suffix.lower() != ".stl":
        report.warnings.append(f"only STL parsing is supported in pure-Python QA (got {path.suffix})")

    try:
        triangles = _parse_stl(path)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"failed to parse mesh: {exc}")
        report.passed = False
        return report

    report.triangles = len(triangles)
    if not triangles:
        report.errors.append("mesh has zero triangles")
        report.passed = False
        return report

    # Bounding box
    report.bounding_box = _bounding_box(triangles)
    if build_volume_mm is not None:
        size = report.bounding_box["size"]
        bv = build_volume_mm
        fits = all(s <= b for s, b in zip(size, bv))
        report.fits_build_volume = fits
        if not fits:
            report.errors.append(
                f"mesh size {size} exceeds build volume {list(bv)}"
            )

    # Edge / manifold / degenerate analysis
    edge_use: Dict[Tuple, int] = {}
    min_edge = float("inf")
    max_edge = 0.0
    degenerate = 0

    for tri in triangles:
        v0, v1, v2 = tri
        if v0 == v1 or v1 == v2 or v0 == v2:
            degenerate += 1
            continue
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            d = _distance(a, b)
            if d < min_edge:
                min_edge = d
            if d > max_edge:
                max_edge = d
            key = _edge_key(a, b)
            edge_use[key] = edge_use.get(key, 0) + 1

    report.degenerate_faces = degenerate
    if min_edge == float("inf"):
        min_edge = 0.0
    report.min_edge_length_mm = round(min_edge, 6)
    report.max_edge_length_mm = round(max_edge, 6)

    open_edges = sum(1 for count in edge_use.values() if count != 2)
    report.open_edges = open_edges
    report.is_manifold = open_edges == 0

    # Aggregate gates
    if not report.is_manifold:
        report.errors.append(f"{open_edges} non-manifold edge(s)")
    if degenerate:
        report.warnings.append(f"{degenerate} degenerate face(s)")
    if min_edge and min_edge < min_wall_thickness_mm:
        report.warnings.append(
            f"shortest edge {min_edge:.3f}mm is below min wall thickness "
            f"{min_wall_thickness_mm:.3f}mm"
        )

    report.passed = not report.errors
    return report
