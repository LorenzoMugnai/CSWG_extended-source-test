"""Plotting utilities for AIRS focal-plane line detection and aperture photometry."""

import itertools

import numpy as np
import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from src.line_detect import detect_line_regions, assign_lines_to_channels


_PEAK_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#8dd3c7",
    "#fb8072",
]


def _wl_array(wl):
    """Return wavelength as a plain numpy array in µm."""
    return wl.to(u.um).value if hasattr(wl, "to") else np.asarray(wl)


def _annotate_foreground_spectrum(ax, foreground):
    """Fill each spectral-line peak with a distinct colour and annotate it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The foreground-spectrum axes.
    foreground : astropy table
        Table with columns ``wavelength`` (µm) and ``radiance`` (W/m²/sr/µm).

    Returns
    -------
    list of dict
        One entry per detected peak with keys ``wl``, ``rad``, ``color``.
    """
    wl = np.asarray(foreground["wavelength"])
    rad = np.asarray(foreground["radiance"])
    dw = np.asarray(foreground["bin_width"])

    if rad.max() == 0:
        return []

    threshold = rad.max() * 1e-4

    # find contiguous non-zero regions (each = one spectral line)
    in_peak = rad > threshold
    change = np.diff(in_peak.astype(int), prepend=0, append=0)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]  # exclusive

    peak_info = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        color = _PEAK_COLORS[i % len(_PEAK_COLORS)]
        idx_peak = s + np.argmax(rad[s:e])
        wl_peak = wl[idx_peak]
        rad_peak = rad[idx_peak]
        # integrated radiance over the bin: W/m²/sr/µm × µm = W/m²/sr
        rad_int = np.sum(rad[s:e] * dw[s:e])

        # extend fill by a few pixels on each side for visibility
        fill_l = max(0, s - 2)
        fill_r = min(len(wl) - 1, e + 2)
        ax.fill_between(
            wl[fill_l : fill_r + 1],
            0,
            rad[fill_l : fill_r + 1],
            color=color,
            alpha=0.25,
            zorder=2,
        )

        # stagger annotation height to avoid overlaps (alternating)
        y_offset = 30 if i % 2 == 0 else 55
        ax.annotate(
            f"{wl_peak:.2f} µm\n{rad_peak:.2e} W/m²/sr/µm\n{rad_int:.2e} W/m²/sr",
            xy=(wl_peak, rad_peak),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=1.2),
            bbox=dict(
                facecolor="white",
                alpha=0.80,
                boxstyle="round,pad=0.25",
                edgecolor=color,
                linewidth=1.2,
            ),
        )
        peak_info.append(
            {"wl": wl_peak, "rad": rad_peak, "rad_int": rad_int, "color": color}
        )

    return peak_info


def _annotate_region(ax, img, wl_v, left, right):
    """Draw ↔ arrow, wavelength width, pixel width, and centre vline on *ax*."""
    ny = img.shape[0]
    wl_l = wl_v[left] if left < len(wl_v) else float("nan")
    wl_r = wl_v[right] if right < len(wl_v) else float("nan")
    width_wl = abs(wl_r - wl_l)
    center = 0.5 * (left + right)
    ax.annotate(
        "",
        xy=(right, ny * 0.85),
        xytext=(left, ny * 0.85),
        arrowprops=dict(arrowstyle="<->", color="k", lw=2, mutation_scale=18),
    )
    ax.text(center, ny * 0.87, f"{width_wl:.3f} µm", color="k", ha="center", fontsize=8)
    ax.text(
        center, ny * 0.70, f"{right - left + 1} px", color="k", ha="center", fontsize=8
    )
    ax.axvline(center, color="red", linestyle="--")


def _detect_and_annotate(
    ax_img,
    img,
    wl,
    search_img=None,
    threshold_factor=0.05,
    ax_profile=None,
    regions=None,
    profile_fill_color=None,
):
    """Annotate *regions* on *ax_img*; detects if *regions* is not provided."""
    wl_v = _wl_array(wl)
    if regions is None:
        src = search_img if search_img is not None else img
        regions = detect_line_regions(src, threshold_factor=threshold_factor)

    profile = np.mean(img, axis=0)
    x = np.arange(len(profile))
    if ax_profile is not None:
        ax_profile.plot(x, profile, color="black")

    for r in regions:
        _annotate_region(ax_img, img, wl_v, r["left"], r["right"])
        if ax_profile is not None:
            ax_profile.axvline(r["center"], color="red", linestyle="--")
            # shade area under the profile for this aperture
            if profile_fill_color is not None:
                mask = (x >= r["left"]) & (x <= r["right"])
                ax_profile.fill_between(
                    x,
                    0,
                    profile,
                    where=mask,
                    interpolate=True,
                    color=profile_fill_color,
                    alpha=0.25,
                )


def _add_wl_axis(ax, wl, nx):
    """Add secondary x-axis with wavelength labels."""
    wl_v = _wl_array(wl)
    ax.set_xlim(0, nx - 1)
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xlabel("Wavelength (µm)")
    ticks = np.linspace(0, nx - 1, 6).astype(int)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([f"{wl_v[p]:.2f}" for p in ticks])


def _overlay_apertures(ax, masks, table=None):
    """Draw white dashed rectangles for each aperture with net_flux label."""
    for i, mask in enumerate(masks):
        ys, xs = np.where(mask.astype(bool))
        if xs.size == 0:
            continue
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        rect = mpatches.Rectangle(
            (left, top),
            right - left + 1,
            bottom - top + 1,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(rect)
        if table is not None and i < len(table):
            if "raw_median" in table.colnames:
                med = float(table["raw_median"][i])
                std = float(table["raw_std"][i])
                text = f"med: {med:.1e}\n± {std:.1e} adu"
            elif "net_flux" in table.colnames:
                text = f"net: {float(table['net_flux'][i]):.3e} adu"
            else:
                text = f"sum: {float(table['raw_sum'][i]):.3e} adu"
            if "snr" in table.colnames:
                snr_val = float(table["snr"][i])
                if not np.isnan(snr_val):
                    text += f"\nSNR: {snr_val:.1f}"
            ax.text(
                0.5 * (left + right),
                0.5 * (top + bottom),
                text,
                color="k",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, boxstyle="round"),
            )


def _plot_channel(
    ax_img,
    ax_prof,
    img,
    wl,
    search_img,
    regions,
    masks,
    table,
    title,
    threshold_factor,
    units,
    profile_fill_color,
    manual_wl_min=None,
    manual_wl_max=None,
    saturation_limit=None,
):
    """Render a single channel focal-plane image with apertures and profile."""
    import matplotlib.pyplot as plt

    ny, nx = img.shape

    cmap = plt.cm.viridis.copy()
    saturated = saturation_limit is not None and float(img.max()) > float(
        saturation_limit
    )
    if saturated:
        cmap.set_over("red")
        vmax = float(saturation_limit)
        extend = "max"
    else:
        vmax = float(img.max())
        extend = "neither"

    im = ax_img.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmax=vmax)
    ax_img.set_title(title, fontsize=10)
    ax_img.set_xlabel("X Pixel", fontsize=9)
    ax_img.set_ylabel("Y Pixel", fontsize=9)
    cb = plt.colorbar(
        im,
        ax=ax_img,
        label=f"Intensity [{units}]" if units else "Intensity",
        extend=extend,
    )
    cb.ax.tick_params(labelsize=8)
    if saturated:
        cb.ax.annotate(
            f"sat: {saturation_limit:.0f}",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )
    _add_wl_axis(ax_img, wl, nx)
    _detect_and_annotate(
        ax_img,
        img,
        wl,
        search_img=search_img,
        threshold_factor=threshold_factor,
        ax_profile=ax_prof,
        regions=regions,
        profile_fill_color=profile_fill_color,
    )
    wl_v = _wl_array(wl)

    # If any pixels exceed the saturation limit, mark the level on the
    # horizontal profile panel to make saturation obvious to the reader.
    if saturation_limit is not None and saturated and ax_prof is not None:
        sat_val = float(saturation_limit)
        ax_prof.axhline(sat_val, color="red", linestyle="--", linewidth=1.2)
        # place a short label at the right edge of the profile plot
        try:
            xlim = ax_prof.get_xlim()
            ax_prof.text(
                xlim[1],
                sat_val,
                f"sat {saturation_limit:.0f}",
                color="red",
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
        except Exception:
            # in case axis limits are not yet set, skip the text annotation
            pass
    if manual_wl_min is not None:
        val = (
            manual_wl_min.to(u.um).value
            if hasattr(manual_wl_min, "to")
            else float(manual_wl_min)
        )
        idx = int(np.argmin(np.abs(wl_v - val)))
        ax_img.axvline(idx, color="white", linestyle=":", linewidth=1.2)
        ax_img.text(
            idx,
            ny * 0.96,
            f"{val:.2f} µm",
            color="white",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax_prof.axvline(idx, color="k", linestyle=":", linewidth=1.2)
    if manual_wl_max is not None:
        val = (
            manual_wl_max.to(u.um).value
            if hasattr(manual_wl_max, "to")
            else float(manual_wl_max)
        )
        idx = int(np.argmin(np.abs(wl_v - val)))
        ax_img.axvline(idx, color="white", linestyle=":", linewidth=1.2)
        ax_img.text(
            idx,
            ny * 0.96,
            f"{val:.2f} µm",
            color="white",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax_prof.axvline(idx, color="k", linestyle=":", linewidth=1.2)
    if masks:
        _overlay_apertures(ax_img, masks, table)
    ax_prof.set_xlabel("X pixel", fontsize=8)
    ax_prof.set_ylabel("Mean intensity", fontsize=8)


# Colour palette used for individual apertures (one colour per line).
_APERTURE_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]


def _plot_aperture_distributions(
    ax_frg,
    ax_bkg,
    img_frg,
    img_bkg,
    masks,
    wl,
    title_frg="FRG — pixel distributions per aperture",
    title_bkg="BKG — pixel distributions per aperture",
):
    """Two side-by-side panels: FRG (left) and BKG (right).

    All apertures are overlaid on each panel; the same colour is used for
    the same aperture across both panels for direct comparison.  Dashed
    vertical lines mark the per-aperture median.
    """
    wl_v = _wl_array(wl)

    for i, mask in enumerate(masks or []):
        color = _APERTURE_COLORS[i % len(_APERTURE_COLORS)]

        _, xs = np.where(mask)
        cx = 0.5 * (int(xs.min()) + int(xs.max())) if xs.size else 0
        wl_c = float(wl_v[min(int(round(cx)), len(wl_v) - 1)])
        lbl = f"{wl_c:.2f} µm"

        vals_frg = img_frg[mask].ravel()
        vals_frg = vals_frg[vals_frg > 0]  # exclude dead / zero-padded pixels
        if vals_frg.size == 0:
            continue
        q25, q75 = np.nanpercentile(vals_frg, [25, 75])
        iqr = q75 - q25
        lo_frg = max(q25 - 3.0 * iqr, vals_frg.min())
        hi_frg = min(q75 + 3.0 * iqr, vals_frg.max())
        ax_frg.hist(
            vals_frg,
            bins=np.linspace(lo_frg, hi_frg, 40),
            color=color,
            alpha=0.55,
            label=lbl,
            density=True,
            histtype="stepfilled",
        )
        ax_frg.axvline(
            float(np.nanmedian(vals_frg)), color=color, lw=1.5, linestyle="--"
        )

        if img_bkg is not None and ax_bkg is not None:
            vals_bkg = img_bkg[mask].ravel()
            vals_bkg = vals_bkg[vals_bkg > 0]  # exclude dead / zero-padded pixels
            if vals_bkg.size == 0:
                continue
            q25b, q75b = np.nanpercentile(vals_bkg, [25, 75])
            iqrb = q75b - q25b
            lo_bkg = max(q25b - 3.0 * iqrb, vals_bkg.min())
            hi_bkg = min(q75b + 3.0 * iqrb, vals_bkg.max())
            ax_bkg.hist(
                vals_bkg,
                bins=np.linspace(lo_bkg, hi_bkg, 40),
                color=color,
                alpha=0.55,
                label=lbl,
                density=True,
                histtype="stepfilled",
            )
            ax_bkg.axvline(
                float(np.nanmedian(vals_bkg)), color=color, lw=1.5, linestyle="--"
            )

    for ax, ttl in [(ax_frg, title_frg)] + (
        [(ax_bkg, title_bkg)] if ax_bkg is not None else []
    ):
        ax.set_xlabel("ADU", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)


def plot(
    ch0_frg,
    ch1_frg,
    wl_ch0,
    wl_ch1,
    foreground=None,
    search_ch0=None,
    search_ch1=None,
    masks_ch0=None,
    masks_ch1=None,
    table_ch0=None,
    table_ch1=None,
    regions_ch0=None,
    regions_ch1=None,
    threshold_factor=0.05,
    units="",
    manual_wl_min_ch0=None,
    manual_wl_max_ch0=None,
    manual_wl_min_ch1=None,
    manual_wl_max_ch1=None,
    ch0_bkg=None,
    ch1_bkg=None,
    table_ch0_bkg=None,
    table_ch1_bkg=None,
    saturation_limit_ch0=None,
    saturation_limit_ch1=None,
    title=None,
):
    """Plot foreground spectra and CH0/CH1 focal planes with detected lines and apertures.

    Parameters
    ----------
    title : str, optional
        Overall figure title.
    ch0_frg, ch1_frg : 2D ndarray
        Focal-plane images for AIRS-CH0 and AIRS-CH1.
    wl_ch0, wl_ch1 : Quantity or array
        Spectral wavelength arrays (µm) for each channel.
    foreground : astropy table, optional
        Single foreground table with columns 'wavelength' and 'radiance'.
    search_ch0, search_ch1 : 2D ndarray, optional
        Foreground-filtered images used for line detection (defaults to ch?_frg).
    masks_ch0, masks_ch1 : list of bool 2D arrays, optional
        Aperture masks from `build_segment_apertures`.
    table_ch0, table_ch1 : QTable, optional
        Photometry tables (must have 'net_flux' or 'raw_sum' column).
    threshold_factor : float
        Detection threshold passed to `detect_line_regions`.
    units : str
        Unit label for the colorbar.
    ch0_bkg, ch1_bkg : 2D ndarray, optional
        Background focal-plane images; if provided, plotted below each channel.
    table_ch0_bkg, table_ch1_bkg : QTable, optional
        Background photometry tables.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    plt.rcParams.update({"font.size": 9})

    has_bkg = (ch0_bkg is not None) or (ch1_bkg is not None)

    if has_bkg:
        # outer grid: spectrum | ch0(frg,fp,bkg,bp,dist) | ch1(frg,fp,bkg,bp,dist)
        height_ratios = [2.0, 2.0, 0.6, 2.0, 0.6, 2.2, 5.0, 0.8, 5.0, 0.8, 2.2]
        fig = plt.figure(figsize=(16, 32))
        gs = GridSpec(11, 1, figure=fig, height_ratios=height_ratios, hspace=0.55)
        ax_spec = fig.add_subplot(gs[0])
        ax_ch0 = fig.add_subplot(gs[1])
        ax_ch0p = fig.add_subplot(gs[2])
        ax_ch0_bkg = fig.add_subplot(gs[3])
        ax_ch0_bkgp = fig.add_subplot(gs[4])
        gs_ch0_dist = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5], wspace=0.35)
        ax_ch0_dist_frg = fig.add_subplot(gs_ch0_dist[0, 0])
        ax_ch0_dist_bkg = fig.add_subplot(gs_ch0_dist[0, 1])
        ax_ch1 = fig.add_subplot(gs[6])
        ax_ch1p = fig.add_subplot(gs[7])
        ax_ch1_bkg = fig.add_subplot(gs[8])
        ax_ch1_bkgp = fig.add_subplot(gs[9])
        gs_ch1_dist = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[10], wspace=0.35)
        ax_ch1_dist_frg = fig.add_subplot(gs_ch1_dist[0, 0])
        ax_ch1_dist_bkg = fig.add_subplot(gs_ch1_dist[0, 1])
        all_single_axes = [
            ax_spec,
            ax_ch0,
            ax_ch0p,
            ax_ch0_bkg,
            ax_ch0_bkgp,
            ax_ch1,
            ax_ch1p,
            ax_ch1_bkg,
            ax_ch1_bkgp,
        ]
    else:
        height_ratios = [2.0, 2.0, 0.6, 2.2, 5.0, 0.8, 2.2]
        fig = plt.figure(figsize=(16, 22))
        gs = GridSpec(7, 1, figure=fig, height_ratios=height_ratios, hspace=0.55)
        ax_spec = fig.add_subplot(gs[0])
        ax_ch0 = fig.add_subplot(gs[1])
        ax_ch0p = fig.add_subplot(gs[2])
        gs_ch0_dist = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.35)
        ax_ch0_dist_frg = fig.add_subplot(gs_ch0_dist[0, 0])
        ax_ch0_dist_bkg = fig.add_subplot(gs_ch0_dist[0, 1])
        ax_ch1 = fig.add_subplot(gs[4])
        ax_ch1p = fig.add_subplot(gs[5])
        gs_ch1_dist = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6], wspace=0.35)
        ax_ch1_dist_frg = fig.add_subplot(gs_ch1_dist[0, 0])
        ax_ch1_dist_bkg = fig.add_subplot(gs_ch1_dist[0, 1])
        ax_ch0_bkg = ax_ch0_bkgp = ax_ch1_bkg = ax_ch1_bkgp = None
        all_single_axes = [ax_spec, ax_ch0, ax_ch0p, ax_ch1, ax_ch1p]

    for a in all_single_axes:
        a.tick_params(labelsize=8)

    # Foreground spectrum
    if foreground is not None:
        ax_spec.plot(
            foreground["wavelength"],
            foreground["radiance"],
            color="k",
            label="foreground",
            zorder=1,
            lw=1,
        )
        _annotate_foreground_spectrum(ax_spec, foreground)
    ax_spec.legend(fontsize=8)
    ax_spec.set_xlabel("Wavelength (µm)", fontsize=9)
    ax_spec.set_ylabel("Radiance [W/m²/sr/µm]", fontsize=9)
    ax_spec.set_title("Foreground Spectrum", fontsize=10)

    # AIRS-CH0 signal
    _plot_channel(
        ax_ch0,
        ax_ch0p,
        ch0_frg,
        wl_ch0,
        search_ch0,
        regions_ch0,
        masks_ch0,
        table_ch0,
        "AIRS-CH0 FRG Focal Plane",
        threshold_factor,
        units,
        "tab:orange",
        manual_wl_min_ch0,
        manual_wl_max_ch0,
        saturation_limit=saturation_limit_ch0,
    )
    ax_ch0p.set_title("Horizontal profile — CH0 FRG", fontsize=9)

    # AIRS-CH0 background
    if ax_ch0_bkg is not None and ch0_bkg is not None:
        _plot_channel(
            ax_ch0_bkg,
            ax_ch0_bkgp,
            ch0_bkg,
            wl_ch0,
            None,
            regions_ch0,
            masks_ch0,
            table_ch0_bkg,
            "AIRS-CH0 BKG Focal Plane",
            threshold_factor,
            units,
            "tab:purple",
            manual_wl_min_ch0,
            manual_wl_max_ch0,
            saturation_limit=saturation_limit_ch0,
        )
        ax_ch0_bkgp.set_title("Horizontal profile — CH0 BKG", fontsize=9)

    # AIRS-CH0 per-aperture distributions
    _plot_aperture_distributions(
        ax_ch0_dist_frg,
        ax_ch0_dist_bkg,
        ch0_frg,
        ch0_bkg,
        masks_ch0,
        wl_ch0,
        title_frg="CH0 — FRG pixel distributions",
        title_bkg="CH0 — BKG pixel distributions",
    )

    # AIRS-CH1 signal
    _plot_channel(
        ax_ch1,
        ax_ch1p,
        ch1_frg,
        wl_ch1,
        search_ch1,
        regions_ch1,
        masks_ch1,
        table_ch1,
        "AIRS-CH1 FRG Focal Plane",
        threshold_factor,
        units,
        "tab:green",
        manual_wl_min_ch1,
        manual_wl_max_ch1,
        saturation_limit=saturation_limit_ch1,
    )
    ax_ch1p.set_title("Horizontal profile — CH1 FRG", fontsize=9)

    # AIRS-CH1 background
    if ax_ch1_bkg is not None and ch1_bkg is not None:
        _plot_channel(
            ax_ch1_bkg,
            ax_ch1_bkgp,
            ch1_bkg,
            wl_ch1,
            None,
            regions_ch1,
            masks_ch1,
            table_ch1_bkg,
            "AIRS-CH1 BKG Focal Plane",
            threshold_factor,
            units,
            "tab:red",
            manual_wl_min_ch1,
            manual_wl_max_ch1,
            saturation_limit=saturation_limit_ch1,
        )
        ax_ch1_bkgp.set_title("Horizontal profile — CH1 BKG", fontsize=9)

    # AIRS-CH1 per-aperture distributions
    _plot_aperture_distributions(
        ax_ch1_dist_frg,
        ax_ch1_dist_bkg,
        ch1_frg,
        ch1_bkg,
        masks_ch1,
        wl_ch1,
        title_frg="CH1 — FRG pixel distributions",
        title_bkg="CH1 — BKG pixel distributions",
    )

    if title is not None:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle("AIRS Focal Planes — Aperture Photometry", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("plots/airs_line_comb.png")
