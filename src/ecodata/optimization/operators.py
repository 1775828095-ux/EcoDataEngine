from __future__ import annotations

import numpy as np
from scipy import sparse

from ecodata.core.exceptions import SchemaError


def aggregation_operator(coarse_shape: tuple[int, int], factor: int) -> sparse.csr_matrix:
    if factor < 1:
        raise SchemaError("Refinement factor must be positive")
    cy, cx = coarse_shape
    fy, fx = cy * factor, cx * factor
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    weight = 1.0 / (factor * factor)
    for iy in range(cy):
        for ix in range(cx):
            coarse_index = iy * cx + ix
            for dy in range(factor):
                for dx in range(factor):
                    fine_index = (iy * factor + dy) * fx + (ix * factor + dx)
                    rows.append(coarse_index)
                    cols.append(fine_index)
                    values.append(weight)
    return sparse.csr_matrix((values, (rows, cols)), shape=(cy * cx, fy * fx))


def difference_operator(shape: tuple[int, int]) -> tuple[sparse.csr_matrix, np.ndarray]:
    ny, nx = shape
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    edge_cells: list[tuple[int, int]] = []
    row = 0
    for iy in range(ny):
        for ix in range(nx):
            current = iy * nx + ix
            if ix + 1 < nx:
                rows.extend([row, row])
                cols.extend([current, current + 1])
                values.extend([-1.0, 1.0])
                edge_cells.append((current, current + 1))
                row += 1
            if iy + 1 < ny:
                rows.extend([row, row])
                cols.extend([current, current + nx])
                values.extend([-1.0, 1.0])
                edge_cells.append((current, current + nx))
                row += 1
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(row, ny * nx))
    return matrix, np.asarray(edge_cells, dtype=np.int64)


def station_operator(
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    ny, nx = len(fine_lat), len(fine_lon)
    lat_index = np.abs(fine_lat[:, None] - station_lat[None, :]).argmin(axis=0)
    lon_index = np.abs(fine_lon[:, None] - station_lon[None, :]).argmin(axis=0)
    cells = lat_index * nx + lon_index
    rows = np.arange(len(cells))
    matrix = sparse.csr_matrix((np.ones(len(cells)), (rows, cells)), shape=(len(cells), ny * nx))
    return matrix, cells

