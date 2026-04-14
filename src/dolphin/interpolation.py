from __future__ import annotations

import logging

import numba
import numpy as np
from numpy.typing import ArrayLike

from .similarity import get_circle_idxs

logger = logging.getLogger("dolphin")


def interpolate(
    ifg: ArrayLike,
    weights: ArrayLike,
    weight_cutoff: float = 0.5,
    num_neighbors: int = 20,
    max_radius: int = 51,
    min_radius: int = 0,
    alpha: float = 0.75,
    tile_size: int = 1024,
) -> np.ndarray:
    """Interpolate a complex interferogram based on pixel weights.

    Build upon persistent scatterer interpolation used in
    [@Chen2015PersistentScattererInterpolation] and
    [@Wang2022AccuratePersistentScatterer] by allowing floating-point weights
    instead of 0/1 PS weights.

    Parameters
    ----------
    ifg : np.ndarray, 2D complex array
        wrapped interferogram to interpolate
    weights : 2D float array
        Array of weights from 0 to 1 indicating how strongly to weigh
        the ifg values when interpolating.
        A special case of this is a PS mask where
            weights[i,j] = True if radar pixel (i,j) is a PS
            weights[i,j] = False if radar pixel (i,j) is not a PS
        Can also pass a coherence image to use as weights.
    weight_cutoff: float
        Threshold to use on `weights` so that pixels where
        `weight[i, j] < weight_cutoff` have phase values replaced by
        an interpolated value.
        The default is 0.5: pixels with weight less than 0.5 are replaced with a
        smoothed version of the surrounding pixels.
    num_neighbors: int (optional)
        number of nearest PS pixels used for interpolation
        num_neighbors = 20 by default
    max_radius : int (optional)
        maximum radius (in pixels) for PS searching
        max_radius = 51 by default
    min_radius : int (optional)
        minimum radius (in pixels) for PS searching
        max_radius = 0 by default
    alpha : float (optional)
        hyperparameter controlling the weight of PS in interpolation: smaller
        alpha means more weight is assigned to PS closer to the access pixel.
        alpha = 0.75 by default
    tile_size : int, optional
        Size of tiles to process at a time to reduce peak memory usage.
        Each tile is expanded by `max_radius` on all sides for context,
        then the interior result is written back.
        Default is 512.

    Returns
    -------
    interpolated_ifg : 2D complex array
        interpolated interferogram with the same amplitude, but different
        wrapped phase at non-ps pixels.

    """
    ifg = np.asarray(ifg)
    nrow, ncol = ifg.shape

    # Avoid a copy if weights is already float32 in [0, 1]
    if weights.dtype == np.float32:
        weights_float = np.asarray(weights)
    else:
        weights_float = np.asarray(weights, dtype=np.float32)
    if np.any(weights_float > 1):
        logger.warning("weights array has values greater than 1. Clipping to 1.")
        weights_float = np.clip(weights_float, 0, 1)
    elif np.any(weights_float < 0):
        logger.warning("weights array has negative values. Clipping to 0.")
        weights_float = np.clip(weights_float, 0, 1)

    indices = np.array(
        get_circle_idxs(max_radius, min_radius=min_radius, sort_output=False)
    )

    interpolated_ifg = np.zeros((nrow, ncol), dtype=np.complex64)

    # Process in tiles to bound peak memory. Each tile is padded by max_radius
    # for context; only the interior (non-padded) result is written to output.
    for r_start in range(0, nrow, tile_size):
        for c_start in range(0, ncol, tile_size):
            r_end = min(r_start + tile_size, nrow)
            c_end = min(c_start + tile_size, ncol)

            # Padded read extents (clamped to image bounds)
            r0p = max(r_start - max_radius, 0)
            c0p = max(c_start - max_radius, 0)
            r1p = min(r_end + max_radius, nrow)
            c1p = min(c_end + max_radius, ncol)

            ifg_tile = ifg[r0p:r1p, c0p:c1p]
            w_tile = weights_float[r0p:r1p, c0p:c1p]
            valid_tile = ifg_tile != 0

            out_tile = np.zeros(ifg_tile.shape, dtype=np.complex64)
            _interp_loop(
                ifg_tile,
                w_tile,
                weight_cutoff,
                valid_tile,
                num_neighbors,
                alpha,
                indices,
                out_tile,
            )

            # Write only the interior (non-padded) region back
            ir0 = r_start - r0p
            ic0 = c_start - c0p
            ir1 = ir0 + (r_end - r_start)
            ic1 = ic0 + (c_end - c_start)
            interpolated_ifg[r_start:r_end, c_start:c_end] = out_tile[ir0:ir1, ic0:ic1]

    return interpolated_ifg


@numba.njit(parallel=True)
def _interp_loop(
    ifg,
    weights,
    weight_cutoff,
    ifg_is_valid_mask,
    num_neighbors,
    alpha,
    indices,
    interpolated_ifg,
):
    nrow, ncol = weights.shape
    nindices = len(indices)
    for r0 in numba.prange(nrow):
        for c0 in range(ncol):
            if not ifg_is_valid_mask[r0, c0]:
                continue
            if weights[r0, c0] >= weight_cutoff:
                interpolated_ifg[r0, c0] = ifg[r0, c0]
                continue

            csum = 0.0 + 0j
            counter = 0
            r2 = np.zeros(num_neighbors, dtype=np.float64)
            cphase = np.zeros(num_neighbors, dtype=np.complex128)

            for i in range(nindices):
                idx = indices[i]
                r = r0 + idx[0]
                c = c0 + idx[1]

                if (
                    (r >= 0)
                    and (r < nrow)
                    and (c >= 0)
                    and (c < ncol)
                    and weights[r, c] >= weight_cutoff
                ):
                    # calculate the square distance to the center pixel
                    r2[counter] = idx[0] ** 2 + idx[1] ** 2

                    cphase[counter] = np.exp(1j * np.angle(ifg[r, c]))
                    counter += 1
                    if counter >= num_neighbors:
                        break

            # `counter` got up to one more than the number of elements
            # The last one will be the largest radius
            r2_norm = (r2[counter - 1] ** alpha) / 2
            for i in range(counter):
                csum += np.exp(-r2[i] / r2_norm) * cphase[i]

            interpolated_ifg[r0, c0] = np.abs(ifg[r0, c0]) * np.exp(1j * np.angle(csum))
