import copy
import numpy as np


def detect_line_regions(img, threshold_factor=0.05, min_width_pixels=1):
    """
    Restituisce lista di dict {left,right,center,width_px,peak} per ogni regione
    contigua del profilo orizzontale sopra soglia.
    """
    profile = np.mean(img, axis=0)
    maxv = np.max(profile)
    if maxv <= 0:
        return []
    mask = profile > (maxv * float(threshold_factor))
    if not mask.any():
        return []
    indices = np.where(mask)[0]
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    regions = []
    for g in groups:
        if len(g) < int(min_width_pixels):
            continue
        left = int(g[0])
        right = int(g[-1])
        center = 0.5 * (left + right)
        peak = np.max(profile[left : right + 1])
        regions.append({"left": left, "right": right, "center": center, "width_px": right - left + 1, "peak": peak})
    return regions


def regions_from_centers_and_widths(centers, widths, nx):
    """Create region dicts from lists of pixel centers and widths.

    Parameters
    ----------
    centers : sequence of numbers
        Pixel centres (x) for each line.
    widths : sequence of numbers
        Widths in pixels (inclusive) for each line.
    nx : int
        Number of spectral pixels (used to clip boundaries).

    Returns
    -------
    list of dict
        Regions with keys `left`, `right`, `center`, `width_px`, `peak`.
    """
    regs = []
    for c, w in zip(centers, widths):
        center = int(round(float(c)))
        w_int = max(1, int(round(float(w))))

        # Compute left such that the resulting inclusive width equals w_int.
        # Prefer centering the region on `center`; adjust at edges if needed.
        left = center - (w_int // 2)
        right = left + w_int - 1

        if left < 0:
            left = 0
            right = left + w_int - 1
        if right > nx - 1:
            right = nx - 1
            left = right - w_int + 1
            if left < 0:
                left = 0

        regs.append({
            "left": int(left),
            "right": int(right),
            "center": 0.5 * (left + right),
            "width_px": int(right - left + 1),
            "peak": None,
        })
    return regs


def regions_from_left_right(pairs, nx):
    """Create region dicts from iterable of (left, right) pixel pairs."""
    regs = []
    for l, r in pairs:
        left = max(0, int(l))
        right = min(nx - 1, int(r))
        regs.append({
            "left": left,
            "right": right,
            "center": 0.5 * (left + right),
            "width_px": right - left + 1,
            "peak": None,
        })
    return regs


def assign_lines_to_channels(foreground, wl_ch0, wl_ch1, buffer_pixels=1):
    """Assign lines from a single foreground table to CH0 and CH1 channels.

    Given a single foreground table containing emission lines across the
    full AIRS band, this function returns two foreground tables: one
    containing only the lines that fall within the CH0 spectral grid
    (with a ±pixel buffer) and one containing only the lines that fall
    within the CH1 spectral grid.

    Lines in the overlap region (approximately 3.119 - 4.071 um) are
    included in both tables, which is physically correct because both
    channels can see those wavelengths.

    Parameters
    ----------
    foreground : `~astropy.table.QTable`
        Foreground table with ``wavelength`` and ``radiance`` columns.
    wl_ch0, wl_ch1 : `~astropy.units.Quantity` or array
        Wavelength arrays for AIRS-CH0 and AIRS-CH1.
    buffer_pixels : int
        Number of pixels to extend the channel coverage on each side
        to account for PSF spill-over (default 1).

    Returns
    -------
    foreground_ch0, foreground_ch1 : `~astropy.table.QTable`
        Two copies of the input table with radiances set to zero
        outside the respective channel coverage.
    """
    import astropy.units as u
    from astropy.table import QTable

    wl0 = np.asarray(wl_ch0.to(u.um).value if hasattr(wl_ch0, "to") else wl_ch0)
    wl1 = np.asarray(wl_ch1.to(u.um).value if hasattr(wl_ch1, "to") else wl_ch1)

    # Pixel scale estimate (mean spacing) for each channel.
    dwl0 = np.mean(np.diff(wl0)) if len(wl0) > 1 else 0.01
    dwl1 = np.mean(np.diff(wl1)) if len(wl1) > 1 else 0.01

    # Extend coverage by buffer_pixels on each side.
    min0, max0 = wl0.min() - buffer_pixels * dwl0, wl0.max() + buffer_pixels * dwl0
    min1, max1 = wl1.min() - buffer_pixels * dwl1, wl1.max() + buffer_pixels * dwl1

    fg_wl = np.asarray(
        foreground["wavelength"].to(u.um).value
        if hasattr(foreground["wavelength"], "to")
        else foreground["wavelength"]
    )

    # Boolean masks for lines inside each channel.
    in0 = (fg_wl >= min0) & (fg_wl <= max0)
    in1 = (fg_wl >= min1) & (fg_wl <= max1)

    # Build per-channel tables by zeroing radiances outside the channel.
    fg0 = foreground.copy()
    fg0["radiance"][~in0] = 0.0

    fg1 = foreground.copy()
    fg1["radiance"][~in1] = 0.0

    return fg0, fg1


def detect_and_build_regions(img, wl, foreground_ch, manual_wl_min=None, manual_wl_max=None,
                             manual_widths=None, manual_centers=None,
                             threshold_factor=0.05, fraction=0.1,
                             buffer_pixels=1, verbose=False):
    """High-level helper to detect line centres from a per-channel foreground
    table and build final regions (optionally using manual widths).

    Parameters
    ----------
    img, wl : ndarray, Quantity/array
        Focal-plane image and spectral solution for the channel.
    foreground_ch : astropy Table
        Per-channel foreground table (radiances zeroed outside channel).
    manual_wl_min, manual_wl_max : float or Quantity or None
        Optional wavelength window to restrict detection (inclusive, µm).
    manual_widths : scalar or sequence or None
        If provided, used as exact pixel widths for all detected centres
        (applied after detection). If None, expansion by profile is used.
    threshold_factor, fraction : float
        Passed to the low-level detection and expansion routines.
    buffer_pixels : int
        Pixel buffer when mapping foreground wavelengths to columns.

    Returns
    -------
    regions : list of dict
        Region dicts ready for aperture construction.
    detected : list of dict
        Raw detected narrow regions (from detect_line_regions) before expansion.
    """
    import astropy.units as u

    wl_v = wl.to(u.um).value if hasattr(wl, 'to') else np.asarray(wl)
    nx = len(wl_v)

    # If user provided manual centres, build regions directly and skip detection
    if manual_centers is not None:
        centers = list(manual_centers) if hasattr(manual_centers, '__len__') else [manual_centers]
        nx = len(wl_v)
        widths = manual_widths if manual_widths is not None else 21
        widths = list(widths) if hasattr(widths, '__len__') else [widths] * len(centers)
        if len(widths) == 1 and len(centers) > 1:
            widths = widths * len(centers)
        regions = regions_from_centers_and_widths(centers, widths, nx)
        if verbose:
            print(f"manual centres provided, built regions: {[{'center':c,'width':w} for c,w in zip(centers,widths)]}")
        return regions, []

    # Map foreground table to columns (±1 pixel) or use manual wl window
    if manual_wl_min is not None or manual_wl_max is not None:
        mn = float(manual_wl_min.to(u.um).value) if hasattr(manual_wl_min, 'to') else (float(manual_wl_min) if manual_wl_min is not None else wl_v.min())
        mx = float(manual_wl_max.to(u.um).value) if hasattr(manual_wl_max, 'to') else (float(manual_wl_max) if manual_wl_max is not None else wl_v.max())
        cols = np.where((wl_v >= mn) & (wl_v <= mx))[0]
        cols = np.unique(np.clip(np.concatenate([cols - 1, cols, cols + 1]), 0, nx - 1)) if cols.size else np.arange(nx)
    else:
        try:
            rad = np.asarray(foreground_ch['radiance'])
        except Exception:
            rad = np.zeros_like(wl_v)
        idxs = np.where(rad > 0)[0]
        if idxs.size == 0:
            cols = np.arange(nx)
        else:
            mapped = [int(np.argmin(np.abs(wl_v - foreground_ch['wavelength'][i].to(u.um).value))) for i in idxs]
            mapped = np.unique(mapped)
            cols = np.unique(np.clip(np.concatenate([mapped - 1, mapped, mapped + 1]), 0, nx - 1))

    if verbose:
        print(f"mapped cols (n={len(cols)}): {cols if len(cols)<50 else '...'}")

    # Build search image and run narrow detection
    search = np.zeros_like(img)
    if cols.size:
        search[:, cols] = img[:, cols]

    detected = detect_line_regions(search, threshold_factor=threshold_factor)
    if verbose:
        print(f"raw detected (n={len(detected)}): {[int(round(r['center'])) for r in detected]}")

    # Optionally filter detected centers by manual wl bounds
    if manual_wl_min is not None or manual_wl_max is not None:
        mn_v = float(manual_wl_min.to(u.um).value) if hasattr(manual_wl_min, 'to') else (float(manual_wl_min) if manual_wl_min is not None else wl_v.min())
        mx_v = float(manual_wl_max.to(u.um).value) if hasattr(manual_wl_max, 'to') else (float(manual_wl_max) if manual_wl_max is not None else wl_v.max())
        kept = []
        for r in detected:
            c = int(round(r['center']))
            if c < 0 or c >= nx:
                continue
            w = float(wl_v[c])
            if w >= mn_v and w <= mx_v:
                kept.append(r)
        detected = kept

    centers = [int(round(r['center'])) for r in detected]
    if verbose:
        wl_centers = [wl_v[c] if 0 <= c < nx else float('nan') for c in centers]
        print(f"detected centers -> pixels: {centers}")
        print(f"detected centers -> wl (um): {wl_centers}")

    # If manual widths provided, build exact regions; otherwise expand by profile
    if manual_widths is not None:
        widths = list(manual_widths) if hasattr(manual_widths, '__len__') else [manual_widths]
        if len(widths) == 1:
            widths = widths * max(1, len(centers))
        if len(widths) != len(centers):
            raise ValueError('manual_widths length must match detected lines or be scalar')
        regions = regions_from_centers_and_widths(centers, widths, nx)
    else:
        regions = expand_to_profile_width(detected, img, fraction=fraction)

    return regions, detected


def expand_to_profile_width(regions, img, fraction=0.1, line_buffer=5, min_width=3):
    """Expand narrow detected regions to their actual PSF footprint in *img*.

    For each region, estimates the local continuum background with a linear
    interpolation between the two nearest inter-line reference pixels (one on
    each side, *line_buffer* columns outside the initial narrow detection).
    Then walks outward from the centre until the excess above this local
    continuum drops below *fraction* × peak_excess.  A minimum width of
    *min_width* pixels is always enforced.

    Using a per-line local interpolation (rather than a global polynomial fit)
    correctly handles steep spectral background gradients such as the rapidly
    rising zodiacal emission at the red edge of AIRS-CH0.

    Parameters
    ----------
    regions : list of dict
        Output of ``detect_line_regions``.
    img : 2-D ndarray
        Full focal-plane image (not the filtered search image).
    fraction : float
        Threshold fraction of the peak excess that defines the region edge.
    line_buffer : int
        Columns between the initial narrow detection edge and the reference
        pixel used for background interpolation on each side.  Default 5.
    min_width : int
        Minimum aperture width in pixels.  Default 3.

    Returns
    -------
    list of dict
        Same keys as input, with updated ``left``, ``right``, ``center``,
        ``width_px``.
    """
    profile = np.mean(img, axis=0)
    nx = len(profile)

    expanded = []
    for r in regions:
        center = int(round(r["center"]))
        center = max(0, min(center, nx - 1))

        half = max(1, int(round((r["right"] - r["left"]) / 2)))

        # Reference pixels just outside the initial narrow detection on each side.
        x_lo = max(0,      center - half - line_buffer)
        x_hi = min(nx - 1, center + half + line_buffer)

        bkg_lo = profile[x_lo]
        bkg_hi = profile[x_hi]

        # Linear background model between the two reference pixels.
        span = x_hi - x_lo
        slope = (bkg_hi - bkg_lo) / span if span > 0 else 0.0
        intercept = bkg_lo - slope * x_lo

        def _bkg(px, _s=slope, _b=intercept):
            return _s * px + _b

        peak = max(0.0, profile[center] - _bkg(center))
        if peak <= 0.0:
            expanded.append(r)
            continue
        thresh = peak * float(fraction)

        left = center
        while left > 0 and max(0.0, profile[left - 1] - _bkg(left - 1)) >= thresh:
            left -= 1

        right = center
        while right < nx - 1 and max(0.0, profile[right + 1] - _bkg(right + 1)) >= thresh:
            right += 1

        # Enforce minimum aperture width.
        if right - left + 1 < min_width:
            pad = (min_width - (right - left + 1) + 1) // 2
            left  = max(0,      left  - pad)
            right = min(nx - 1, right + pad)

        expanded.append({
            "left": left,
            "right": right,
            "center": 0.5 * (left + right),
            "width_px": right - left + 1,
            "peak": float(profile[center] - _bkg(center)),
        })
    return expanded
