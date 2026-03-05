import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sps
import scipy.interpolate as spi

import read_data_results3 as rd
import Read_spectrum as rdsp


# -----------------------------
# Helpers: FT pipeline
# -----------------------------

def center_zpd(x, y):
    """
    Shift interferogram so the centerburst (max |y|) is at the middle index.
    Also shift x so that ZPD corresponds to x=0.
    """
    i0 = int(np.argmax(np.abs(y)))
    shift = (len(y) // 2) - i0
    y2 = np.roll(y, shift)
    x2 = x - x[i0]
    return x2, y2


def ft_spectrum_from_interferogram(
    x_opd_m,
    y,
    N=2**19,
    window="blackmanharris",
    detrend_type="linear",
    interp_kind="linear",
    use_real_spectrum=True,
    clip_negative=True,
):
    """
    Convert interferogram y(x) sampled at OPD positions (meters) into a spectrum.

    Returns:
      nu        : axis in cycles/m (from rfftfreq)
      intensity : spectrum on nu axis (real-part or magnitude)
      meta      : dict with diagnostics
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_opd_m, dtype=float)

    # Detrend
    y = sps.detrend(y, type=detrend_type)

    # Center ZPD explicitly
    x, y = center_zpd(x, y)

    # Window/apodization
    if window == "hann":
        w = np.hanning(len(y))
    elif window == "blackmanharris":
        w = sps.windows.blackmanharris(len(y))
    elif window == "kaiser14":
        w = sps.windows.kaiser(len(y), beta=14.0)
    else:
        raise ValueError("window must be one of: 'hann', 'blackmanharris', 'kaiser14'")
    y = y * w

    # Resample onto uniform OPD grid (FFT requires uniform sampling)
    xs = np.linspace(x.min(), x.max(), int(N))
    f = spi.interp1d(x, y, kind=interp_kind, fill_value="extrapolate")
    ys = f(xs)

    dx = xs[1] - xs[0]

    # Real FFT
    Y = np.fft.rfft(ys)
    nu = np.fft.rfftfreq(len(xs), d=dx)  # cycles per meter

    # Drop DC
    mask = nu > 0
    nu = nu[mask]
    Y = Y[mask]

    if use_real_spectrum:
        intensity = np.real(Y)
        if clip_negative:
            intensity = intensity.copy()
            intensity[intensity < 0] = 0.0
    else:
        intensity = np.abs(Y)

    OPDmax = float(np.max(np.abs(xs)))
    expected_dnu = (1.0 / (2.0 * OPDmax)) if OPDmax > 0 else np.nan

    meta = {
        "dx_m": float(dx),
        "OPDmax_m": OPDmax,
        "expected_dnu_cycles_per_m": float(expected_dnu),
        "N_uniform": int(len(xs)),
    }

    return nu, intensity, meta


def data_i(
    file,
    metres_per_microstep=3.66e-11,
    N=2**19,
    window="blackmanharris",
    use_real_spectrum=True,
):
    """
    Load interferogram from your 'read_data3' format and compute FT spectrum.
    Returns: [name, nu_axis, intensity, meta]
    """
    results = rd.read_data3("data/" + file + ".txt")

    y1 = np.array(results[1], dtype=float)
    x_microsteps = np.array(results[5], dtype=float)
    x_opd_m = x_microsteps * metres_per_microstep

    nu, intensity, meta = ft_spectrum_from_interferogram(
        x_opd_m,
        y1,
        N=N,
        window=window,
        detrend_type="linear",
        interp_kind="linear",
        use_real_spectrum=use_real_spectrum,
        clip_negative=True,
    )

    return [file, nu, intensity, meta]


def data_g(file):
    results = rdsp.read_data4("data/" + file + ".txt")
    return [file, results[0], results[1]]


# -----------------------------
# Artifact quantification
# -----------------------------

def quantify_artifacts(
    x_axis,
    intensity,
    smooth_window=None,
    poly=3,
    prominence_mode="sigma",  # "frac" or "sigma"
    prominence_frac=0.02,
    prominence_sigma=3.0,
    band=None,               # (xmin, xmax) in x_axis units
):
    """
    Subtract smooth envelope and detect peaks in residual.
    Returns banded arrays too, so plotting cannot mismatch lengths.

    Returns:
      rms_ratio, x_used, y_used, env, residual, peak_table
    """
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(intensity, dtype=float)

    # Sort ascending x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Band limit (optional)
    if band is not None:
        m = (x >= band[0]) & (x <= band[1])
        x = x[m]
        y = y[m]

    # Normalize y to max=1
    y = y / np.max(y) if np.max(y) != 0 else y

    # Choose smoothing window adaptively if not provided
    if smooth_window is None:
        smooth_window = max(51, (len(y) // 50) | 1)  # ~2% of points, odd
    # Ensure valid odd window
    smooth_window = min(smooth_window, len(y) - (1 - len(y) % 2))
    if smooth_window % 2 == 0:
        smooth_window -= 1
    smooth_window = max(poly + 2 + (poly + 2) % 2, smooth_window)  # ensure > poly

    env = sps.savgol_filter(y, smooth_window, poly)
    residual = y - env

    # Peak threshold
    if prominence_mode == "frac":
        prom = prominence_frac  # since y is normalized
    elif prominence_mode == "sigma":
        prom = float(prominence_sigma * np.std(residual))
    else:
        raise ValueError("prominence_mode must be 'frac' or 'sigma'")

    peaks, props = sps.find_peaks(residual, prominence=prom)

    artifact_rms = float(np.sqrt(np.mean(residual**2)))
    signal_rms = float(np.sqrt(np.mean(y**2))) if np.mean(y**2) > 0 else np.nan
    rms_ratio = artifact_rms / signal_rms if signal_rms and not np.isnan(signal_rms) else np.nan

    peak_table = {
        "x_axis": x[peaks],
        "residual_height": residual[peaks],
        "prominence": props.get("prominences", np.array([])),
    }

    return rms_ratio, x, y, env, residual, peak_table


# -----------------------------
# Plotting
# -----------------------------

def to_wavelength(x_nu):
    """Convert nu (cycles/m) -> wavelength (m) safely."""
    x_nu = np.asarray(x_nu, dtype=float)
    m = x_nu > 0
    wl = np.empty_like(x_nu)
    wl[:] = np.nan
    wl[m] = 1.0 / x_nu[m]
    return wl


def plot_spectrum(nu, intensity, label, display_as_wavelength=True):
    y = np.asarray(intensity, dtype=float)
    y = y / np.max(y) if np.max(y) != 0 else y

    if display_as_wavelength:
        wl = to_wavelength(nu)
        m = np.isfinite(wl)
        plt.plot(wl[m], y[m], label=label)
        plt.xlabel("Wavelength (m)  [display: 1/nu]")
    else:
        plt.plot(nu, y, label=label)
        plt.xlabel("nu (cycles/m)")
    plt.ylabel("Normalized Intensity")


def plot_artifacts(x_used, y_used, env, residual, peak_table, title, display_as_wavelength=True):
    plt.figure(title)
    plt.title(title)

    if display_as_wavelength:
        wl = to_wavelength(x_used)
        m = np.isfinite(wl)
        # wavelength decreases with nu increasing, so sort by wl for nice plot
        order = np.argsort(wl[m])
        xx = wl[m][order]
        yy = y_used[m][order]
        ee = env[m][order]
        rr = residual[m][order]

        plt.plot(xx, yy, label="Normalized signal")
        plt.plot(xx, ee, label="Envelope (SavGol)")
        plt.plot(xx, rr, label="Residual")

        if len(peak_table["x_axis"]):
            pk_wl = to_wavelength(peak_table["x_axis"])
            pk_m = np.isfinite(pk_wl)
            pk_wl = pk_wl[pk_m]
            # interpolate residual for marker y values
            r_at_pk = np.interp(pk_wl, xx, rr)
            plt.scatter(pk_wl, r_at_pk, marker="x", label="Residual peaks")
        plt.xlabel("Wavelength (m)  [display: 1/nu]")
    else:
        plt.plot(x_used, y_used, label="Normalized signal")
        plt.plot(x_used, env, label="Envelope (SavGol)")
        plt.plot(x_used, residual, label="Residual")

        if len(peak_table["x_axis"]):
            r_at_pk = np.interp(peak_table["x_axis"], x_used, residual)
            plt.scatter(peak_table["x_axis"], r_at_pk, marker="x", label="Residual peaks")
        plt.xlabel("nu (cycles/m)")

    plt.grid(True)
    plt.legend()
    plt.ylabel("Normalized / Residual")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Compute FT spectrum from interferogram
    name, nu, intensity, meta = data_i(
        "white_light_4",
        metres_per_microstep=3.66e-11,
        N=2**19,
        window="blackmanharris",
        use_real_spectrum=True,  # try True; set False to compare abs()
    )

    print("Diagnostics:", meta)

    # Quantify artifacts in nu-space (recommended)
    nu_min = np.quantile(nu, 0.10)
    nu_max = np.quantile(nu, 0.90)

    rms_ratio, x_used, y_used, env, residual, peak_table = quantify_artifacts(
        nu,
        intensity,
        smooth_window=None,          # adaptive
        poly=3,
        prominence_mode="sigma",
        prominence_sigma=3.0,
        band=(nu_min, nu_max),
    )

    # Print top 10 peaks by prominence
    order = np.argsort(peak_table["prominence"])[::-1]
    print("\nTop residual peaks (nu-space):")
    for i in order[:10]:
        print("nu =", peak_table["x_axis"][i], "prom =", peak_table["prominence"][i])

    print("\nArtifact RMS / Signal RMS:", rms_ratio)

    # Plot spectrum (display as wavelength for familiarity)
    plt.figure("Spectrum (displayed vs wavelength)")
    plt.title("FT Spectrum (computed in nu, displayed as wavelength=1/nu)")
    plot_spectrum(nu, intensity, label=name, display_as_wavelength=True)
    plt.xlim(3.5e-7, 8e-7)
    plt.grid(True)
    plt.legend()

    # Plot artifact diagnostic using the SAME arrays used in quantification (no mismatch)
    plot_artifacts(
        x_used, y_used, env, residual, peak_table,
        title="Artifacts (nu-space analysis, displayed as wavelength)",
        display_as_wavelength=True
    )
    plt.xlim(3.5e-7, 8e-7)

    plt.show()