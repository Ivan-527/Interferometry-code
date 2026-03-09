# -*- coding: utf-8 -*-
"""
FT spectrum analysis with:
1) correct OPD scaling
2) wavelength-domain conversion with Jacobian
3) visible-band artifact estimate
4) etalon linearity test
5) only the important plots

Important plots produced:
- FT spectrum in wavelength domain
- FT vs grating comparison
- Etalon linearity test
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sps
import scipy.interpolate as spi

import read_data_results3 as rd
import Read_spectrum as rdsp


# -----------------------------
# Core FT helpers
# -----------------------------

def center_zpd(x, y):
    """
    Shift interferogram so the centerburst is at the middle index.
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
    use_real_spectrum=False,
    clip_negative=True,
):
    """
    Convert interferogram y(x) sampled vs OPD (m) into FT spectrum.

    Returns:
        nu        : cycles / m
        intensity : spectrum vs nu
        meta      : diagnostics dict
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_opd_m, dtype=float)

    # Detrend and center ZPD
    y = sps.detrend(y, type=detrend_type)
    x, y = center_zpd(x, y)

    # Apodization
    if window == "hann":
        w = np.hanning(len(y))
    elif window == "blackmanharris":
        w = sps.windows.blackmanharris(len(y))
    elif window == "kaiser14":
        w = sps.windows.kaiser(len(y), beta=14.0)
    else:
        raise ValueError("window must be 'hann', 'blackmanharris', or 'kaiser14'")
    y = y * w

    # Uniform OPD grid
    xs = np.linspace(x.min(), x.max(), int(N))
    f = spi.interp1d(x, y, kind=interp_kind, fill_value="extrapolate")
    ys = f(xs)

    dx = xs[1] - xs[0]

    # FFT
    Y = np.fft.rfft(ys)
    nu = np.fft.rfftfreq(len(xs), d=dx)

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
    expected_dnu = 1.0 / (2.0 * OPDmax) if OPDmax > 0 else np.nan

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
    use_real_spectrum=False,
):
    """
    Load interferogram data and compute FT spectrum.

    metres_per_microstep is MIRROR displacement, so multiply by 2 for OPD.
    """
    results = rd.read_data3("data/" + file + ".txt")

    y1 = np.array(results[1], dtype=float)
    x_microsteps = np.array(results[5], dtype=float)

    # Mirror displacement -> OPD
    x_opd_m = 2.0 * x_microsteps * metres_per_microstep

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
    """
    Load grating spectrum.
    Assumes rdsp.read_data4 returns [wavelength, intensity].
    """
    results = rdsp.read_data4("data/" + file + ".txt")
    return [file, np.asarray(results[0], dtype=float), np.asarray(results[1], dtype=float)]


# -----------------------------
# Spectrum conversion
# -----------------------------

def nu_to_wavelength_spectrum(nu_axis, intensity_nu):
    """
    Convert spectrum from nu-space to wavelength-space including Jacobian.

    If I_nu dnu = I_lambda dlambda and nu = 1/lambda, then:
        I_lambda = I_nu / lambda^2
    """
    nu_axis = np.asarray(nu_axis, dtype=float)
    intensity_nu = np.asarray(intensity_nu, dtype=float)

    m = np.isfinite(nu_axis) & np.isfinite(intensity_nu) & (nu_axis > 0)
    nu = nu_axis[m]
    I_nu = intensity_nu[m]

    wl = 1.0 / nu
    I_wl = I_nu / (wl ** 2)

    idx = np.argsort(wl)
    return wl[idx], I_wl[idx]


# -----------------------------
# Diagnostics
# -----------------------------

def print_resolution_estimates(OPDmax_m, wavelengths_m=(5e-7, 6e-7, 7e-7)):
    delta_sigma_m_inv = 1.0 / (2.0 * OPDmax_m)
    delta_sigma_cm_inv = delta_sigma_m_inv * 0.01

    print("Estimated FT resolution:")
    print("Δσ =", delta_sigma_m_inv, "m^-1")
    print("Δσ =", delta_sigma_cm_inv, "cm^-1")

    for wl in wavelengths_m:
        delta_lambda_m = (wl ** 2) * delta_sigma_m_inv
        delta_lambda_nm = delta_lambda_m * 1e9
        print(f"At λ = {wl:.3e} m, Δλ ≈ {delta_lambda_m:.3e} m = {delta_lambda_nm:.4f} nm")


def quantify_artifacts(
    x_axis,
    intensity,
    smooth_window=None,
    poly=3,
    prominence_sigma=3.0,
):
    """
    Estimate residual ripple in a band by subtracting a smooth envelope.
    """
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(intensity, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if np.max(y) != 0:
        y = y / np.max(y)

    if smooth_window is None:
        smooth_window = max(51, (len(y) // 40) | 1)

    smooth_window = min(smooth_window, len(y) - (1 - len(y) % 2))
    if smooth_window % 2 == 0:
        smooth_window -= 1
    if smooth_window <= poly:
        smooth_window = poly + 3 if (poly + 3) % 2 == 1 else poly + 4

    env = sps.savgol_filter(y, smooth_window, poly)
    residual = y - env

    prom = float(prominence_sigma * np.std(residual))
    peaks, props = sps.find_peaks(residual, prominence=prom, width=3)

    artifact_rms = float(np.sqrt(np.mean(residual**2)))
    signal_rms = float(np.sqrt(np.mean(y**2)))
    rms_ratio = artifact_rms / signal_rms if signal_rms > 0 else np.nan

    peak_table = {
        "x_axis": x[peaks],
        "residual_height": residual[peaks],
        "prominence": props.get("prominences", np.array([])),
    }

    return rms_ratio, x, y, env, residual, peak_table


def smooth_curve_nm(wl, y, smooth_nm=30e-9, poly=3):
    """
    Smooth a wavelength-domain curve using a wavelength-scale Savitzky-Golay window.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(wl) & np.isfinite(y)
    wl = wl[m]
    y = y[m]

    idx = np.argsort(wl)
    wl = wl[idx]
    y = y[idx]

    dw = np.mean(np.diff(wl))
    window = int(smooth_nm / dw)
    window = max(window, 101)
    if window % 2 == 0:
        window += 1
    if window >= len(wl):
        window = len(wl) - 1
        if window % 2 == 0:
            window -= 1

    y_smooth = sps.savgol_filter(y, window, poly)
    return wl, y_smooth


# -----------------------------
# Etalon test
# -----------------------------

def etalon_linearity_test(wl, intensity, wl_band=(5e-7, 7e-7), prominence=0.05):
    """
    Test etalon relation:
        1/lambda_m = a * m + b

    Returns:
        wl_peaks, inv_wl_peaks, peak_numbers, fit_coeffs, fitted_inv_wl, r2
    """
    wl = np.asarray(wl, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    m = np.isfinite(wl) & np.isfinite(intensity)
    wl = wl[m]
    intensity = intensity[m]

    idx = np.argsort(wl)
    wl = wl[idx]
    intensity = intensity[idx]

    m_band = (wl >= wl_band[0]) & (wl <= wl_band[1])
    wl = wl[m_band]
    intensity = intensity[m_band]

    if np.max(intensity) != 0:
        intensity = intensity / np.max(intensity)

    peaks, _ = sps.find_peaks(intensity, prominence=prominence)

    wl_peaks = wl[peaks]
    inv_wl_peaks = 1.0 / wl_peaks
    peak_numbers = np.arange(len(wl_peaks), dtype=float)

    if len(wl_peaks) < 3:
        print("Not enough peaks for etalon linearity test.")
        return wl_peaks, inv_wl_peaks, peak_numbers, None, None, np.nan

    fit_coeffs = np.polyfit(peak_numbers, inv_wl_peaks, 1)
    fitted_inv_wl = np.polyval(fit_coeffs, peak_numbers)

    ss_res = np.sum((inv_wl_peaks - fitted_inv_wl) ** 2)
    ss_tot = np.sum((inv_wl_peaks - np.mean(inv_wl_peaks)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    print("\nEtalon linearity test:")
    print("Number of detected peaks:", len(wl_peaks))
    print("Fit slope:", fit_coeffs[0])
    print("Fit intercept:", fit_coeffs[1])
    print("R^2 =", r2)

    return wl_peaks, inv_wl_peaks, peak_numbers, fit_coeffs, fitted_inv_wl, r2


def estimate_etalon_thickness_from_fit(fit_coeffs, n=1.5):
    """
    slope = 1 / (2 n d), using absolute slope for physical thickness.
    """
    if fit_coeffs is None:
        return np.nan

    slope = abs(fit_coeffs[0])
    if slope == 0:
        return np.nan

    d = 1.0 / (2.0 * n * slope)
    print("Estimated etalon thickness from fit:")
    print("d =", d, "m")
    print("d =", d * 1e6, "microns")
    return d


# -----------------------------
# Response correction
# -----------------------------

def estimate_response_envelope(ft_wl, ft_intensity, gr_wl, gr_intensity,
                               wl_band=(5e-7, 7e-7), smooth_nm=30e-9):
    """
    Estimate slow FT system response relative to grating spectrum.

    Returns:
        wl, ft_raw, gr_raw, ft_env, gr_env, response_env, ft_corrected
    """
    ft_wl = np.asarray(ft_wl, dtype=float)
    ft_intensity = np.asarray(ft_intensity, dtype=float)
    gr_wl = np.asarray(gr_wl, dtype=float)
    gr_intensity = np.asarray(gr_intensity, dtype=float)

    m_ft = np.isfinite(ft_wl) & np.isfinite(ft_intensity)
    ft_wl = ft_wl[m_ft]
    ft_intensity = ft_intensity[m_ft]

    m_gr = np.isfinite(gr_wl) & np.isfinite(gr_intensity)
    gr_wl = gr_wl[m_gr]
    gr_intensity = gr_intensity[m_gr]

    i_ft = np.argsort(ft_wl)
    ft_wl = ft_wl[i_ft]
    ft_intensity = ft_intensity[i_ft]

    i_gr = np.argsort(gr_wl)
    gr_wl = gr_wl[i_gr]
    gr_intensity = gr_intensity[i_gr]

    m_band = (ft_wl >= wl_band[0]) & (ft_wl <= wl_band[1])
    wl = ft_wl[m_band]
    ft = ft_intensity[m_band]

    gr = np.interp(wl, gr_wl, gr_intensity)

    if np.max(ft) != 0:
        ft = ft / np.max(ft)
    if np.max(gr) != 0:
        gr = gr / np.max(gr)

    _, ft_env = smooth_curve_nm(wl, ft, smooth_nm=smooth_nm, poly=3)
    _, gr_env = smooth_curve_nm(wl, gr, smooth_nm=smooth_nm, poly=3)

    eps = 1e-12
    response_env = gr_env / (ft_env + eps)

    if np.max(response_env) != 0:
        response_env = response_env / np.max(response_env)

    ft_corrected = ft * response_env
    if np.max(ft_corrected) != 0:
        ft_corrected = ft_corrected / np.max(ft_corrected)

    return wl, ft, gr, ft_env, gr_env, response_env, ft_corrected


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    # -------- Settings --------
    ft_file = "white_light_3"
    grating_file = "White_LED_Lens"
    metres_per_microstep = 3.77835e-11   # mirror displacement
    visible_band = (5e-7, 7e-7)
    response_smooth_nm = 30e-9
    # --------------------------

    # FT spectrum
    name, nu, intensity_nu, meta = data_i(
        ft_file,
        metres_per_microstep=metres_per_microstep,
        N=2**19,
        window="blackmanharris",
        use_real_spectrum=False,
    )

    print("OPDmax (mm):", meta["OPDmax_m"] * 1000)
    print("Diagnostics:", meta)
    print()
    print_resolution_estimates(meta["OPDmax_m"])

    # Convert to wavelength domain
    wl, intensity_wl = nu_to_wavelength_spectrum(nu, intensity_nu)

    # Visible-band artifact estimate
    m_vis = np.isfinite(wl) & (wl >= visible_band[0]) & (wl <= visible_band[1])
    rms_ratio, x_used, y_used, env, residual, peak_table = quantify_artifacts(
        wl[m_vis],
        intensity_wl[m_vis],
        smooth_window=None,
        poly=3,
        prominence_sigma=3.0,
    )

    print("\nArtifact RMS / Signal RMS (visible band):", rms_ratio)
    print("Visible wavelength range used:", np.min(x_used), "to", np.max(x_used))
    print("Residual min/max:", np.min(residual), np.max(residual))

    # Grating spectrum
    gr_name, gr_wl, gr_intensity = data_g(grating_file)

    # Etalon test
    wl_peaks, inv_wl_peaks, peak_numbers, fit_coeffs, fitted_inv_wl, r2 = etalon_linearity_test(
        wl,
        intensity_wl,
        wl_band=visible_band,
        prominence=0.05,
    )
    d_fit = estimate_etalon_thickness_from_fit(fit_coeffs, n=1.5)

    # Response correction
    wl_resp, ft_raw_resp, gr_raw_resp, ft_env_resp, gr_env_resp, response_env, ft_corrected = estimate_response_envelope(
        wl,
        intensity_wl,
        gr_wl,
        gr_intensity,
        wl_band=visible_band,
        smooth_nm=response_smooth_nm,
    )

    # -----------------------------
    # Important plots only
    # -----------------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1) FT spectrum
    y_full = intensity_wl / np.max(intensity_wl) if np.max(intensity_wl) != 0 else intensity_wl
    axes[0].plot(wl, y_full, label=name)
    axes[0].set_xlim(3.5e-7, 8e-7)
    axes[0].set_xlabel("Wavelength (m)")
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_title("FT spectrum (wavelength domain)")
    axes[0].grid(True)
    axes[0].legend()

    # 2) FT vs grating comparison
    axes[1].plot(wl_resp, ft_raw_resp, label=f"{name} (raw FT)")
    axes[1].plot(wl_resp, ft_corrected, label=f"{name} (response-corrected FT)")
    axes[1].plot(wl_resp, gr_raw_resp, label=gr_name)
    axes[1].set_xlim(visible_band[0], visible_band[1])
    axes[1].set_xlabel("Wavelength (m)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].set_title("FT vs grating comparison")
    axes[1].grid(True)
    axes[1].legend()

    # 3) Etalon linearity test
    if len(wl_peaks) > 0:
        axes[2].plot(peak_numbers, inv_wl_peaks, "o", label="Detected peaks")
        if fitted_inv_wl is not None:
            axes[2].plot(peak_numbers, fitted_inv_wl, "-", label=f"Linear fit (R²={r2:.4f})")
    axes[2].set_xlabel("Peak index")
    axes[2].set_ylabel("1 / wavelength (m^-1)")
    axes[2].set_title("Etalon linearity test")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
