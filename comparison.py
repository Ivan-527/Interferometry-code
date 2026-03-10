import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spf
import scipy.interpolate as spi
import scipy.optimize as opt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
import read_data_results3 as rd
import Read_spectrum as rdsp


# =========================================================
# 1. Data loading
# =========================================================

def data(file):
    """Interferometer-derived spectrum: returns [file, wavelength_m, intensity]."""
    results = rd.read_data3('data/' + file + '.txt')

    metres_per_microstep = 3.655e-11

    y1 = np.array(results[1], dtype=float)
    x = np.array(results[5], dtype=float) * metres_per_microstep

    # CHANGED: sort before interpolation / spline
    order = np.argsort(x)
    x = x[order]
    y1 = y1[order]

    y1 = y1 - y1.mean()
    y2 = y1 * np.hanning(len(y1))

    N = 1000
    xs = np.linspace(x[0], x[-1], N)
    ys = spi.CubicSpline(x, y2)(xs)

    dx = xs[1] - xs[0]
    yf = spf.fft(ys)
    xf = spf.fftfreq(len(xs), d=dx)

    xf = spf.fftshift(xf)
    yf = spf.fftshift(yf)

    mask = xf > 0
    freq = xf[mask]
    amp = np.abs(yf[mask])

    wavelength = 1.0 / freq

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    amp = amp[order]

    amp = amp / np.max(amp)

    return [file, wavelength, amp]


def grating(file):
    """Grating-derived spectrum: returns [file, wavelength_m, intensity]."""
    results = rdsp.read_data4('data/' + file + '.txt')

    x = np.array(results[0], dtype=float)
    y = np.array(results[1], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y = y / np.max(y)

    return [file, x, y]


# =========================================================
# 2. Helpers
# =========================================================

def make_valid_savgol_window(n_points, requested_window, polyorder):
    w = int(requested_window)
    if w >= n_points:
        w = n_points - 1
    if w % 2 == 0:
        w -= 1
    if w < polyorder + 2:
        w = polyorder + 3
        if w % 2 == 0:
            w += 1
    if w >= n_points:
        w = n_points - 1
        if w % 2 == 0:
            w -= 1
    if w <= polyorder:
        raise ValueError("Not enough points for Savitzky-Golay filter.")
    return w


def estimate_led_yerr_from_smoothing(lam, y, window=11, polyorder=3, label='LED'):
    """
    Estimate LED spectrum uncertainty from residuals about a smooth trend.
    Returns a wavelength-dependent yerr using:
      yerr = sqrt(local_residual^2 + RMS_residual^2)
    """
    lam = np.array(lam, dtype=float)
    y = np.array(y, dtype=float)

    valid = np.isfinite(lam) & np.isfinite(y)
    lam = lam[valid]
    y = y[valid]

    order = np.argsort(lam)
    lam = lam[order]
    y = y[order]

    if len(y) < 10:
        raise ValueError(f"[{label}] Not enough points for LED yerr estimation.")

    window = make_valid_savgol_window(len(y), window, polyorder)
    smooth = savgol_filter(y, window, polyorder)
    residual = y - smooth
    rms = np.sqrt(np.mean(residual**2))

    yerr = np.sqrt(residual**2 + rms**2)

    print(f"\n[{label}] --- LED yerr summary ---")
    print(f"Savgol window used: {window}")
    print(f"Residual RMS:       {rms:.4e}")
    print(f"Mean LED yerr:      {np.mean(yerr):.4e}")
    print(f"Max LED yerr:       {np.max(yerr):.4e}")

    return {
        'lambda': lam,
        'y': y,
        'smooth': smooth,
        'residual': residual,
        'rms': rms,
        'yerr': yerr
    }


# =========================================================
# 3. Normalise both, suppress bumps, get transmission
# =========================================================

def suppress_bumps_and_get_transmission(
    white_x,
    white_y,
    led_x,
    led_y,
    trend_window=15,
    bump_window=7,
    polyorder=3,
    lam_min=4.3e-7,
    lam_max=6.6e-7,
    label='white_light_4',
    make_plots=True
):
    white_x = np.array(white_x, dtype=float)
    white_y = np.array(white_y, dtype=float)
    led_x = np.array(led_x, dtype=float)
    led_y = np.array(led_y, dtype=float)

    ow = np.argsort(white_x)
    white_x = white_x[ow]
    white_y = white_y[ow]

    ol = np.argsort(led_x)
    led_x = led_x[ol]
    led_y = led_y[ol]

    mask_white = (white_x >= lam_min) & (white_x <= lam_max)
    mask_led = (led_x >= lam_min) & (led_x <= lam_max)

    white_x = white_x[mask_white]
    white_y = white_y[mask_white]

    led_x = led_x[mask_led]
    led_y = led_y[mask_led]

    # NOTE:
    # These are separately max-normalised spectra.
    # The resulting ratio is therefore a relative shape ratio, not an
    # absolute transmission unless independent scaling justifies it.
    white_y = white_y / np.nanmax(white_y)
    led_y = led_y / np.nanmax(led_y)

    led_interp = interp1d(led_x, led_y, bounds_error=False, fill_value=np.nan)
    led_on_white = led_interp(white_x)

    valid = np.isfinite(white_x) & np.isfinite(white_y) & np.isfinite(led_on_white)
    lam = white_x[valid]
    white_norm = white_y[valid]
    led_norm = led_on_white[valid]

    valid_raw = (
        np.isfinite(white_norm) &
        np.isfinite(led_norm) &
        (np.abs(led_norm) > 1e-12)
    )

    lam_raw = lam[valid_raw]
    white_norm_raw = white_norm[valid_raw]
    led_norm_raw = led_norm[valid_raw]
    white_over_led_raw = white_norm_raw / led_norm_raw

    if len(lam) < 10:
        raise ValueError("Not enough overlapping points in chosen wavelength range.")

    residual = white_norm - led_norm

    trend_window = make_valid_savgol_window(len(residual), trend_window, polyorder)
    trend = savgol_filter(residual, trend_window, polyorder)
    detrended = residual - trend

    bump_window = make_valid_savgol_window(len(detrended), bump_window, polyorder)
    bumps = savgol_filter(detrended, bump_window, polyorder)

    white_suppressed = white_norm - bumps

    valid_t = (
        np.isfinite(white_suppressed) &
        np.isfinite(led_norm) &
        (np.abs(white_suppressed) > 1e-12) &
        (np.abs(led_norm) > 1e-12)
    )

    lam_t = lam[valid_t]
    white_suppressed_t = white_suppressed[valid_t]
    led_norm_t = led_norm[valid_t]

    transmission_direct = led_norm_t / white_suppressed_t
    inverse_transmission_direct = white_suppressed_t / led_norm_t

    print(f"\n[{label}] windows used:")
    print("trend_window =", trend_window)
    print("bump_window  =", bump_window)

    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), num=f'{label}: preprocessing')

        axes[0].plot(lam * 1e9, residual, label='Residual')
        axes[0].plot(lam * 1e9, trend, label='Trend')
        axes[0].plot(lam * 1e9, bumps, label='Extracted bumps')
        axes[0].set_title('Residual processing')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(lam * 1e9, white_norm, label='White (norm)')
        axes[1].plot(lam * 1e9, white_suppressed, label='White bumps suppressed')
        axes[1].plot(lam * 1e9, led_norm, label='LED lens (norm)')
        axes[1].set_title('Normalised spectra')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Normalised intensity')
        axes[1].grid()
        axes[1].legend()

        fig.tight_layout()

    return {
        'lambda': lam,
        'white_norm': white_norm,
        'led_norm': led_norm,
        'residual': residual,
        'trend': trend,
        'detrended': detrended,
        'bumps': bumps,
        'white_suppressed': white_suppressed,
        'lambda_transmission': lam_t,
        'white_suppressed_transmission': white_suppressed_t,
        'led_norm_transmission': led_norm_t,
        'transmission': transmission_direct,
        'inverse_transmission': inverse_transmission_direct,
        'raw_ratio_lambda': lam_raw,
        'raw_ratio': white_over_led_raw
    }


# =========================================================
# 4. yerr estimation from cumulative processing uncertainties
# =========================================================

def estimate_yerr_from_processing(
    result,
    secondary_peak_amp=None,
    dominant_peak_amp=None,
    label='white_light_4',
    make_plots=True
):
    """
    Estimate wavelength-dependent yerr for the bump-suppressed spectrum.

    Components combined in quadrature:
    1. Residual fringe contamination:
         |bumps| * (secondary_peak_amp / dominant_peak_amp)
    2. Residual/trend mismatch RMS:
         RMS(residual - trend), applied uniformly
    """

    lam = np.array(result['lambda'], dtype=float)
    white_suppressed = np.array(result['white_suppressed'], dtype=float)
    residual = np.array(result['residual'], dtype=float)
    trend = np.array(result['trend'], dtype=float)
    bumps = np.array(result['bumps'], dtype=float)

    valid = (
        np.isfinite(lam) &
        np.isfinite(white_suppressed) &
        np.isfinite(residual) &
        np.isfinite(trend) &
        np.isfinite(bumps)
    )

    lam = lam[valid]
    white_suppressed = white_suppressed[valid]
    residual = residual[valid]
    trend = trend[valid]
    bumps = bumps[valid]

    if (
        secondary_peak_amp is not None and
        dominant_peak_amp is not None and
        dominant_peak_amp > 0
    ):
        secondary_fraction = secondary_peak_amp / dominant_peak_amp
    else:
        secondary_fraction = 0.0

    residual_fringe_yerr = np.abs(bumps) * secondary_fraction

    scaling_residual = residual - trend
    scaling_rms = np.sqrt(np.mean(scaling_residual**2))
    scaling_yerr = np.full_like(white_suppressed, scaling_rms)

    yerr = np.sqrt(
        residual_fringe_yerr**2 +
        scaling_yerr**2
    )

    mean_frac = np.mean(yerr / np.maximum(np.abs(white_suppressed), 1e-12)) * 100

    print(f"\n[{label}] --- white-spectrum yerr summary ---")
    print(f"Secondary fringe fraction: {secondary_fraction:.4f} ({secondary_fraction*100:.2f}%)")
    print(f"Scaling residual RMS:      {scaling_rms:.4e}")
    print(f"Mean fringe yerr:          {np.mean(residual_fringe_yerr):.4e}")
    print(f"Mean total yerr:           {np.mean(yerr):.4e}")
    print(f"Max total yerr:            {np.max(yerr):.4e}")
    print(f"Mean fractional yerr:      {mean_frac:.2f}%")

    if make_plots:
        norm_factor = np.nanmax(np.abs(white_suppressed))
        if norm_factor <= 0:
            norm_factor = 1.0

        white_suppressed_norm = white_suppressed / norm_factor
        yerr_norm = yerr / norm_factor

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), num=f'{label}: uncertainty')

        axes[0].plot(lam * 1e9, residual_fringe_yerr, label='Residual fringe term')
        axes[0].plot(lam * 1e9, scaling_yerr, linestyle='--', label='Scaling RMS term')
        axes[0].plot(lam * 1e9, yerr, linewidth=2, label='Total yerr')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Uncertainty')
        axes[0].set_title('White-spectrum uncertainty components')
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(lam * 1e9, white_suppressed_norm, label='Bump-suppressed spectrum')
        axes[1].fill_between(
            lam * 1e9,
            white_suppressed_norm - yerr_norm,
            white_suppressed_norm + yerr_norm,
            alpha=0.3,
            label='±1σ band'
        )
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Normalised intensity')
        axes[1].set_title('Bump-suppressed spectrum with uncertainty')
        axes[1].grid()
        axes[1].legend()

        fig.tight_layout()

    return {
        'lambda': lam,
        'yerr': yerr,
        'residual_fringe_yerr': residual_fringe_yerr,
        'scaling_yerr': scaling_yerr,
        'secondary_fraction': secondary_fraction,
        'scaling_rms': scaling_rms
    }


# =========================================================
# 5. Inverse transmission linear fit
#    CHANGED:
#    - weighted fit using sigma=yerr_inverse
#    - xerr_frac = 5.9%
# =========================================================

def analyze_inverse_transmission(
    lam_t,
    transmission,
    yerr_inverse,
    fit_max=6.55e-7,
    cutoff_min=6.45e-7,
    cutoff_max=6.60e-7,
    cutoff_step=0.01e-7,
    xerr_frac=0.059,
    make_plots=True
):
    lam_t = np.array(lam_t, dtype=float)
    transmission = np.array(transmission, dtype=float)
    yerr_inverse = np.array(yerr_inverse, dtype=float)

    valid = (
        np.isfinite(lam_t) &
        np.isfinite(transmission) &
        np.isfinite(yerr_inverse) &
        (np.abs(transmission) > 1e-12) &
        (np.abs(yerr_inverse) > 1e-20)
    )

    lam_t = lam_t[valid]
    transmission = transmission[valid]
    yerr_inverse = yerr_inverse[valid]

    inverse_transmission = 1.0 / transmission

    def line_model(x, m, c):
        return m * x + c

    fit_mask = lam_t <= fit_max
    lam_fit = lam_t[fit_mask]
    y_fit = inverse_transmission[fit_mask]
    sigma_y_fit = yerr_inverse[fit_mask]

    if len(lam_fit) < 3:
        raise ValueError("Not enough points below fit_max for linear fit.")

    p0 = [0.0, np.mean(y_fit)]

    # CHANGED: weighted fit
    popt, pcov = opt.curve_fit(
        line_model,
        lam_fit,
        y_fit,
        sigma=sigma_y_fit,
        absolute_sigma=True,
        p0=p0,
        maxfev=100000
    )

    m, c = popt
    m_err = np.sqrt(pcov[0, 0])
    c_err = np.sqrt(pcov[1, 1])

    line_fit_all = m * lam_t + c
    line_fit_used = m * lam_fit + c
    residuals_used = y_fit - line_fit_used

    sigma_x_fit = xerr_frac * lam_fit
    sigma_y_fit = np.maximum(sigma_y_fit, 1e-12)
    sigma_tot_sq = sigma_y_fit**2 + (m * sigma_x_fit)**2

    chi2 = np.sum((residuals_used**2) / sigma_tot_sq)
    dof = len(y_fit) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan

    weights = 1.0 / sigma_tot_sq
    y_mean_w = np.sum(weights * y_fit) / np.sum(weights)
    ss_res = np.sum(weights * (y_fit - line_fit_used) ** 2)
    ss_tot = np.sum(weights * (y_fit - y_mean_w) ** 2)
    r2_weighted = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print("\nInverse-transmission linear fit")
    print(f"Cutoff used = {fit_max:.3e} m")
    print(f"xerr_frac   = {xerr_frac:.3%}")
    print(f"m = {m:.6e} ± {m_err:.6e}")
    print(f"c = {c:.6e} ± {c_err:.6e}")
    print(f"Chi-squared = {chi2:.6e}")
    print(f"Reduced chi-squared = {chi2_red:.6e}")
    print(f"Weighted R^2 = {r2_weighted:.6f}")

    cutoff_values = np.arange(cutoff_min, cutoff_max + 0.5 * cutoff_step, cutoff_step)

    cutoff_list = []
    m_list = []
    c_list = []
    chi2_red_list = []
    r2_list = []

    for cutoff in cutoff_values:
        mask = lam_t <= cutoff
        x_sub = lam_t[mask]
        y_sub = inverse_transmission[mask]
        sigma_y_sub = yerr_inverse[mask]

        if len(x_sub) < 3:
            continue

        try:
            p_sub, _ = opt.curve_fit(
                line_model,
                x_sub,
                y_sub,
                sigma=sigma_y_sub,
                absolute_sigma=True,
                p0=[0.0, np.mean(y_sub)],
                maxfev=100000
            )
            m_sub, c_sub = p_sub

            y_model_sub = m_sub * x_sub + c_sub

            sigma_x_sub = xerr_frac * x_sub
            sigma_y_sub = np.maximum(sigma_y_sub, 1e-12)

            sigma_tot_sq_sub = sigma_y_sub**2 + (m_sub * sigma_x_sub)**2
            resid_sub = y_sub - y_model_sub

            chi2_sub = np.sum((resid_sub**2) / sigma_tot_sq_sub)
            dof_sub = len(y_sub) - 2
            chi2_red_sub = chi2_sub / dof_sub if dof_sub > 0 else np.nan

            w_sub = 1.0 / sigma_tot_sq_sub
            y_mean_w_sub = np.sum(w_sub * y_sub) / np.sum(w_sub)
            ss_res_sub = np.sum(w_sub * (y_sub - y_model_sub) ** 2)
            ss_tot_sub = np.sum(w_sub * (y_sub - y_mean_w_sub) ** 2)
            r2_sub = 1 - ss_res_sub / ss_tot_sub if ss_tot_sub > 0 else np.nan

            cutoff_list.append(cutoff)
            m_list.append(m_sub)
            c_list.append(c_sub)
            chi2_red_list.append(chi2_red_sub)
            r2_list.append(r2_sub)

        except Exception:
            continue

    cutoff_list = np.array(cutoff_list)
    m_list = np.array(m_list)
    c_list = np.array(c_list)
    chi2_red_list = np.array(chi2_red_list)
    r2_list = np.array(r2_list)

    if make_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), num='Inverse transmission analysis')

        axes[0].errorbar(
            lam_fit * 1e9,
            y_fit,
            xerr=sigma_x_fit * 1e9,
            yerr=sigma_y_fit,
            fmt='o',
            markersize=4,
            capsize=2,
            label='Fit data'
        )
        axes[0].plot(lam_t * 1e9, line_fit_all, label='Weighted linear fit')
        axes[0].axvline(fit_max * 1e9, linestyle='--', color='gray', label='Fit cutoff')
        axes[0].set_title('Inverse transmission with weighted fit')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid()
        axes[0].legend()

        axes[1].errorbar(
            lam_fit * 1e9,
            residuals_used,
            xerr=sigma_x_fit * 1e9,
            yerr=sigma_y_fit,
            fmt='o',
            markersize=4,
            capsize=2,
            label='Residuals'
        )
        axes[1].axhline(0, linestyle='--', color='gray')
        axes[1].set_title('Fit residuals')
        axes[1].set_xlabel('Wavelength (nm)')
        axes[1].set_ylabel('Residual')
        axes[1].grid()
        axes[1].legend()

        axes[2].plot(cutoff_list * 1e9, m_list, marker='o', label='Slope m')
        axes[2].axvline(fit_max * 1e9, linestyle='--', color='gray', label='Chosen cutoff')
        axes[2].set_title('Slope stability vs cutoff')
        axes[2].set_xlabel('Cutoff wavelength (nm)')
        axes[2].set_ylabel('Slope m')
        axes[2].grid()
        axes[2].legend()

        fig.tight_layout()

    return {
        'lambda': lam_t,
        'inverse_transmission': inverse_transmission,
        'line_fit': line_fit_all,
        'residuals_used': residuals_used,
        'lambda_fit': lam_fit,
        'm': m,
        'c': c,
        'm_err': m_err,
        'c_err': c_err,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'r2_weighted': r2_weighted,
        'cutoff_values': cutoff_list,
        'slope_values': m_list,
        'intercept_values': c_list,
        'chi2_red_values': chi2_red_list,
        'r2_values': r2_list,
        'fit_max': fit_max
    }


# =========================================================
# 6. Bump FFT for peak amplitudes, fringe spacing and uncertainties
# =========================================================

def analyse_bump_structure_fft(
    lam,
    bumps,
    n_uniform=2000,
    min_spatial_freq=None,
    fft_peak_prominence_ratio=0.05,
    max_peaks=5,
    label='white_light_4',
    make_plots=True
):
    lam = np.array(lam, dtype=float)
    bumps = np.array(bumps, dtype=float)

    valid = np.isfinite(lam) & np.isfinite(bumps)
    lam = lam[valid]
    bumps = bumps[valid]

    if len(lam) < 10:
        print(f"[{label}] WARNING: Not enough points for bump FFT.")
        return None

    order = np.argsort(lam)
    lam = lam[order]
    bumps = bumps[order]

    lam_uniform = np.linspace(lam[0], lam[-1], n_uniform)
    bumps_uniform = spi.CubicSpline(lam, bumps)(lam_uniform)

    bumps_uniform = bumps_uniform - np.mean(bumps_uniform)
    bumps_windowed = bumps_uniform * np.hanning(len(bumps_uniform))

    dlam = lam_uniform[1] - lam_uniform[0]
    fft_vals = spf.fft(bumps_windowed)
    fft_freq = spf.fftfreq(len(lam_uniform), d=dlam)

    pos = fft_freq > 0
    spatial_freq = fft_freq[pos]
    fft_amp = np.abs(fft_vals[pos])

    if min_spatial_freq is None:
        min_spatial_freq = 1.0 / (lam[-1] - lam[0])

    use = spatial_freq >= min_spatial_freq
    spatial_freq = spatial_freq[use]
    fft_amp = fft_amp[use]

    if len(fft_amp) < 2:
        print(f"[{label}] WARNING: Too few FFT bins after cutoff.")
        return None

    prom = fft_peak_prominence_ratio * np.max(fft_amp)
    peak_indices, _ = find_peaks(fft_amp, prominence=prom)

    if len(peak_indices) < 2:
        peak_indices = np.argsort(fft_amp)[-min(max_peaks, len(fft_amp)):]

    peak_amps = fft_amp[peak_indices]
    peak_freqs = spatial_freq[peak_indices]

    order_pk = np.argsort(peak_amps)
    peak_amps = peak_amps[order_pk]
    peak_freqs = peak_freqs[order_pk]

    dominant_amp = peak_amps[-1]
    secondary_amp = peak_amps[-2] if len(peak_amps) >= 2 else None

    fringe_spacing = 1.0 / spatial_freq
    peak_fringe_spacing = 1.0 / peak_freqs

    # Uncertainty estimate from FFT bin width:
    # df ≈ 1 / total_lambda_span
    total_span = lam_uniform[-1] - lam_uniform[0]
    df = 1.0 / total_span
    peak_freq_err = np.full_like(peak_freqs, 0.5 * df)

    # propagate s = 1/f -> sigma_s = sigma_f / f^2
    peak_fringe_spacing_err = peak_freq_err / (peak_freqs**2)

    print(f"\n[{label}] --- Bump FFT summary ---")
    print(f"FFT bin spacing df:       {df:.6e} cycles/m")
    print(f"Dominant peak amplitude:  {dominant_amp:.6e}")
    if secondary_amp is not None:
        print(f"Secondary peak amplitude: {secondary_amp:.6e}")
        print(f"Secondary/dominant ratio: {secondary_amp/dominant_amp:.4f}")

    print("\nDetected fringe-spacing peaks:")
    for i, (fpk, sfpk, dsfpk, apk) in enumerate(
        zip(peak_freqs, peak_fringe_spacing, peak_fringe_spacing_err, peak_amps), start=1
    ):
        print(
            f"Peak {i}: "
            f"freq = {fpk:.6e} cycles/m, "
            f"Δλ = {sfpk*1e9:.3f} ± {dsfpk*1e9:.3f} nm, "
            f"amp = {apk:.6e}"
        )

    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), num=f'Bump FFT: {label}')

        axes[0].plot(spatial_freq * 1e-6, fft_amp, label='FFT amplitude')
        axes[0].plot(peak_freqs * 1e-6, peak_amps, 'o', color='red', label='Detected peaks')
        axes[0].set_xlabel('Spatial frequency (cycles / µm of wavelength)')
        axes[0].set_ylabel('FFT amplitude')
        axes[0].set_title('FFT in spatial-frequency domain')
        axes[0].grid()
        axes[0].legend()

        axes[1].errorbar(
            peak_fringe_spacing * 1e9,
            peak_amps,
            xerr=peak_fringe_spacing_err * 1e9,
            fmt='o',
            capsize=3,
            label='FFT peaks'
        )
        axes[1].plot(fringe_spacing * 1e9, fft_amp, alpha=0.35, label='FFT amplitude')
        axes[1].set_xlim(0, 50)
        axes[1].set_xlabel('Fringe spacing Δλ (nm)')
        axes[1].set_ylabel('FFT amplitude')
        axes[1].set_title('FFT peaks in fringe-spacing domain')
        axes[1].grid()
        axes[1].legend()

        fig.tight_layout()

    return {
        'spatial_freq': spatial_freq,
        'fft_amp': fft_amp,
        'peak_freqs': peak_freqs,
        'peak_freq_errs': peak_freq_err,
        'peak_amps': peak_amps,
        'peak_fringe_spacing': peak_fringe_spacing,
        'peak_fringe_spacing_err': peak_fringe_spacing_err,
        'dominant_amp': dominant_amp,
        'secondary_amp': secondary_amp,
        'fft_bin_spacing': df
    }


# =========================================================
# 7. Main
# =========================================================

white = data('white_light_4')
led = grating('White_LED_Lens')

white_x = np.array(white[1], dtype=float)
white_y = np.array(white[2], dtype=float)

led_x = np.array(led[1], dtype=float)
led_y = np.array(led[2], dtype=float)

result = suppress_bumps_and_get_transmission(
    white_x,
    white_y,
    led_x,
    led_y,
    trend_window=15,   # will become 15 internally because Savitzky-Golay needs odd window
    bump_window=7,
    polyorder=3,
    lam_min=4.4e-7,
    lam_max=7e-7,
    label='white_light_4',
    make_plots=True
)

bump_fft_result = analyse_bump_structure_fft(
    result['lambda'],
    result['bumps'],
    n_uniform=2000,
    min_spatial_freq=None,
    fft_peak_prominence_ratio=0.05,
    max_peaks=5,
    label='white_light_4',
    make_plots=True
)

if bump_fft_result is not None and bump_fft_result['secondary_amp'] is not None:
    dominant_amp = bump_fft_result['dominant_amp']
    secondary_amp = bump_fft_result['secondary_amp']
else:
    dominant_amp = None
    secondary_amp = None

yerr_result = estimate_yerr_from_processing(
    result,
    secondary_peak_amp=secondary_amp,
    dominant_peak_amp=dominant_amp,
    label='white_light_4',
    make_plots=True
)

# CHANGED: estimate denominator (LED) uncertainty too
led_yerr_result = estimate_led_yerr_from_smoothing(
    led_x,
    led_y,
    window=11,
    polyorder=3,
    label='White_LED_Lens'
)

# Propagate uncertainties for inverse transmission:
# inverse_transmission = white_suppressed / led
#
# sigma_inv^2 ≈ (sigma_white / led)^2 + (white_suppressed * sigma_led / led^2)^2

lam_white = np.array(yerr_result['lambda'], dtype=float)
yerr_white = np.array(yerr_result['yerr'], dtype=float)

lam_led = np.array(led_yerr_result['lambda'], dtype=float)
yerr_led = np.array(led_yerr_result['yerr'], dtype=float)

lam_t = np.array(result['lambda_transmission'], dtype=float)
white_t = np.array(result['white_suppressed_transmission'], dtype=float)
led_t = np.array(result['led_norm_transmission'], dtype=float)

interp_yerr_white = interp1d(lam_white, yerr_white, bounds_error=False, fill_value=np.nan)
interp_yerr_led = interp1d(lam_led, yerr_led, bounds_error=False, fill_value=np.nan)

yerr_white_on_t = interp_yerr_white(lam_t)
yerr_led_on_t = interp_yerr_led(lam_t)

den = np.maximum(np.abs(led_t), 1e-12)

yerr_inverse = np.sqrt(
    (yerr_white_on_t / den)**2 +
    ((np.abs(white_t) * yerr_led_on_t) / (den**2))**2
)

inverse_result = analyze_inverse_transmission(
    result['lambda_transmission'],
    result['transmission'],
    yerr_inverse=yerr_inverse,
    fit_max=6.55e-7,
    cutoff_min=6.45e-7,
    cutoff_max=6.60e-7,
    cutoff_step=0.01e-7,
    xerr_frac=0.0059,   # CHANGED: 5.9%
    make_plots=True
)

plt.show()
