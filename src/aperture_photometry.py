"""Aperture photometry utilities for rectangular segment apertures.

Functions
---------
build_segment_apertures(regions, ny, nx, padding=0)
    Build boolean rectangular masks from detected line regions.
aperture_photometry(img, metas, masks, background="local", annulus_width=5)
    Compute raw, background, and net flux for each aperture.
save_apertures(path, metas, masks, spectral=None)
    Save aperture geometry and masks to a compressed .npz file.
load_apertures(path) -> (metas, masks, spectral_or_None)
    Load apertures saved with save_apertures.
apply_saved_apertures(img, path, background="local", annulus_width=5) -> QTable
    Convenience wrapper: load, resize if needed, run photometry, return QTable.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from astropy.table import QTable


def build_segment_apertures(
    regions: List[Dict], ny: int, nx: int, padding: int = 0
) -> Tuple[List[Dict], List[np.ndarray]]:
    """Build rectangular aperture masks for each detected region.

    Parameters
    ----------
    regions : list of dicts with keys 'left' and 'right' (pixel columns)
    ny, nx  : image dimensions (rows, cols)
    padding : extra pixels added to left and right of each aperture

    Returns
    -------
    metas : list of serialisable geometry dicts
    masks : list of bool 2D arrays matching image shape
    """
    metas, masks = [], []
    for r in regions:
        left = max(0, int(r.get("left", 0)) - int(padding))
        right = min(nx - 1, int(r.get("right", 0)) + int(padding))
        top, bottom = 0, ny - 1

        mask = np.zeros((ny, nx), dtype=bool)
        mask[top : bottom + 1, left : right + 1] = True

        metas.append(
            {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "width_px": right - left + 1,
                "height_px": bottom - top + 1,
                "area_px": int(mask.sum()),
            }
        )
        masks.append(mask)

    return metas, masks


def aperture_photometry(
    img: np.ndarray,
    apertures_meta: List[Dict],
    masks: List[np.ndarray],
    background: str = "local",
    annulus_width: int = 5,
) -> List[Dict]:
    """Compute aperture sums and background-subtracted fluxes.

    Parameters
    ----------
    img             : 2D image array
    apertures_meta  : list of geometry dicts from build_segment_apertures
    masks           : corresponding boolean aperture masks
    background      : 'local' (annulus), 'global' (image median), or 'none'
    annulus_width   : half-width in pixels of the local background annulus

    Returns
    -------
    list of dicts: raw_sum, area_px, bkg_per_pix, bkg_total, net_flux
    """
    ny, nx = img.shape
    global_med = float(np.nanmedian(img)) if background == "global" else None
    results = []

    for meta, mask in zip(apertures_meta, masks):
        area = int(mask.sum())
        pix = img[mask]
        valid = pix[pix > 0]  # exclude dead / zero-padded pixels
        valid = valid if valid.size > 0 else pix  # fallback if all zero
        raw_sum = float(np.nansum(pix))
        raw_median = float(np.nanmedian(valid))
        raw_std = float(np.nanstd(valid))

        if background == "none":
            bkg_per_pix = 0.0
        elif background == "global":
            bkg_per_pix = global_med
        else:
            l = max(0, meta["left"] - annulus_width)
            r = min(nx - 1, meta["right"] + annulus_width)
            annulus = np.zeros((ny, nx), dtype=bool)
            annulus[meta["top"] : meta["bottom"] + 1, l : r + 1] = True
            annulus[mask] = False
            vals = img[annulus]
            bkg_per_pix = float(np.nanmedian(vals)) if vals.size > 0 else 0.0

        bkg_total = bkg_per_pix * area
        results.append(
            {
                "raw_sum": raw_sum,
                "raw_median": raw_median,
                "raw_std": raw_std,
                "area_px": area,
                "bkg_per_pix": bkg_per_pix,
                "bkg_total": bkg_total,
                "net_flux": raw_sum - bkg_total,
            }
        )

    return results


def save_apertures(
    path: str,
    apertures_meta: List[Dict],
    masks: List[np.ndarray],
    spectral: np.ndarray | None = None,
):
    """Save aperture geometry and masks to a compressed .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    stacked = np.stack([m.astype(np.uint8) for m in masks])
    meta_json = json.dumps(apertures_meta)
    kwargs = dict(masks=stacked, meta=np.array(meta_json, dtype="S"))
    if spectral is not None:
        kwargs["wl"] = np.asarray(spectral, dtype=float)
    np.savez_compressed(path, **kwargs)


def load_apertures(path: str) -> Tuple[List[Dict], List[np.ndarray], np.ndarray | None]:
    """Load apertures saved with save_apertures.

    Returns (metas, masks, spectral_or_None).
    """
    with np.load(path, allow_pickle=True) as data:
        stacked = data["masks"]
        scalar = data["meta"].item()
        wl = data["wl"] if "wl" in data else None

    meta_json = (
        scalar.decode() if isinstance(scalar, (bytes, bytearray)) else str(scalar)
    )
    metas = json.loads(meta_json)
    masks = [s.astype(bool) for s in stacked]
    return metas, masks, wl


def apply_saved_apertures(
    img: np.ndarray,
    path: str,
    background: str = "local",
    annulus_width: int = 5,
) -> QTable:
    """Load saved apertures and run photometry on *img*. Returns an Astropy QTable.

    If the saved masks do not match the image shape, they are rebuilt from
    the stored geometry using simple pixel scaling.
    """
    metas, masks, _ = load_apertures(path)

    ny_img, nx_img = img.shape
    if masks and masks[0].shape != (ny_img, nx_img):
        ny_m, nx_m = masks[0].shape
        sx = nx_img / float(nx_m) if nx_m > 0 else 1.0
        sy = ny_img / float(ny_m) if ny_m > 0 else 1.0
        new_masks = []
        for meta in metas:
            l = max(0, min(int(round(meta["left"] * sx)), nx_img - 1))
            r = max(0, min(int(round(meta["right"] * sx)), nx_img - 1))
            t = max(0, min(int(round(meta["top"] * sy)), ny_img - 1))
            b = max(0, min(int(round(meta["bottom"] * sy)), ny_img - 1))
            m = np.zeros((ny_img, nx_img), dtype=bool)
            m[t : b + 1, l : r + 1] = True
            new_masks.append(m)
        masks = new_masks

    results = aperture_photometry(
        img, metas, masks, background=background, annulus_width=annulus_width
    )

    table = QTable()
    for k in ["left", "right", "top", "bottom", "width_px", "height_px", "area_px"]:
        table[k] = [m.get(k) for m in metas]
    table["raw_sum"] = [r["raw_sum"] for r in results]
    table["bkg_per_pix"] = [r["bkg_per_pix"] for r in results]
    table["bkg_total"] = [r["bkg_total"] for r in results]
    table["net_flux"] = [r["net_flux"] for r in results]
    return table
