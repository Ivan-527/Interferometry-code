import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.fftpack as spf
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import read_data_results3 as rd
import Read_spectrum as rdsp


# =========================================================
# 1. Data loading
# =========================================================

def data(file):
    """
    Interferometer-derived spectrum.
    Returns [file, wavelength_m, normalized_intensity]
    """
    results = rd.read_data3('data/' + file + '.txt')

    metres_per_microstep = 3.66e-11  # metres

    y1 = np.array(results[1], dtype=float)
    x = np.array(results[5], dtype=float) * metres_per_microstep

    y1 = y1 - y1.mean()
    y2 = y1 * np.hanning(len(y1))

    # Resample onto regular grid
    N = 1000
    xs = np.linspace(x[0], x[-1], N)
    cs = spi.CubicSpline(x, y2)
    ys = cs(xs)

    dx = xs[1] - xs[0]
    yf1 = spf.fft(ys)
    xf1 = spf.fftfreq(len(xs), d=dx)

    xf1 = spf.fftshift(xf1)
    yf1 = spf.fftshift(yf1)

    mask = xf1 > 0
    freq = xf1[mask]
    amp = np.abs(yf1[mask])

    wavelength = 1.0 / freq

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    amp = amp[order]

    amp = amp / np.max(amp)

    return [file, wavelength, amp]


def grating(file):
    """
    Grating-derived spectrum.
    Returns [file, wavelength_m, normalized_intensity]
    """
    results = rdsp.read_data4('data/' + file + '.txt')

    x = np.array(results[0], dtype=float)
    y = np.array(results[1], dtype=float)

    # Uncomment if your grating wavelengths are in nm:
    # x = x * 1e-9

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y = y / np.max(y)

    return [file, x, y]


# =========================================================
# 2. Benchmark background using White_LED_Lens
#    with quadratic correction
# =========================================================

def benchmark_model(lam, led_interp, a, b, c, d, lam0):
    return a * led_interp + b + c * (lam - lam0) + d * (lam - lam0) ** 2


def fit_led_benchmark_background(
    white_x,
    white_y,
    led_x,
    led_y,
    lam_min=4.3e-7,
    lam_max=6.6e-7,
    label='white_light_4'
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

    mask = (white_x >= lam_min) & (white_x <= lam_max)
    lam = white_x[mask]
    y = white_y[mask]

    if len(lam) < 20:
        raise ValueError("Not enough white_light_4 points in fit range.")

    interp_led = interp1d(
        led_x,
        led_y,
        bounds_error=False,
        fill_value=np.nan
    )

    led_on_white = interp_led(lam)

    valid = np.isfinite(lam) & np.isfinite(y) & np.isfinite(led_on_white)
    lam = lam[valid]
    y = y[valid]
    led_on_white = led_on_white[valid]

    if len(lam) < 20:
        raise ValueError("Not enough overlap between white_light_4 and White_LED_Lens.")

    lam0 = np.mean(lam)

    def fit_func(lam_, a, b, c, d):
        led_vals = interp_led(lam_)
        return benchmark_model(lam_, led_vals, a, b, c, d, lam0)

    p0 = [1.0, 0.0, 0.0, 0.0]

    popt, pcov = opt.curve_fit(fit_func, lam, y, p0=p0, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))

    background_fit = fit_func(lam, *popt)
    residual_fit = y - background_fit

    rss = np.sum(residual_fit ** 2)
    dof = len(y) - len(popt)
    sigma_est = np.sqrt(rss / dof) if dof > 0 else np.nan

    print(f"\n[{label}] --- White_LED_Lens benchmark background fit ---")
    print(f"Fit range: {lam_min:.3e} m to {lam_max:.3e} m")
    print(f"a = {popt[0]:.6e} ± {perr[0]:.6e}")
    print(f"b = {popt[1]:.6e} ± {perr[1]:.6e}")
    print(f"c = {popt[2]:.6e} ± {perr[2]:.6e}")
    print(f"d = {popt[3]:.6e} ± {perr[3]:.6e}")
    print(f"RSS = {rss:.6e}")
    print(f"Estimated common sigma = {sigma_est:.6e}")

    # Full-range background
    led_full = interp_led(white_x)
    finite_full = np.isfinite(white_x) & np.isfinite(white_y) & np.isfinite(led_full)

    white_x_full = white_x[finite_full]
    white_y_full = white_y[finite_full]
    led_full = led_full[finite_full]

    bg_full = benchmark_model(
        white_x_full,
        led_full,
        popt[0],
        popt[1],
        popt[2],
        popt[3],
        lam0
    )

    residual_full = white_y_full - bg_full

    return {
        'lambda_fit': lam,
        'white_fit': y,
        'led_fit': led_on_white,
        'background_fit': background_fit,
        'residual_fit': residual_fit,
        'white_full_x': white_x_full,
        'white_full_y': white_y_full,
        'background_full': bg_full,
        'residual_full': residual_full,
        'params': popt,
        'param_errors': perr,
        'rss': rss,
        'sigma_est': sigma_est
    }


# =========================================================
# 3. Frequency analysis of benchmark residual
# =========================================================

def frequency_analysis_from_benchmark_residual(
    benchmark_result,
    n_uniform=2000,
    min_mod_freq=2.0e-5,
    fft_peak_prominence_ratio=0.05,
    max_peaks=5,
    refractive_index=1.0,
    label='white_light_4'
):
    lam = np.array(benchmark_result['lambda_fit'], dtype=float)
    residual = np.array(benchmark_result['residual_fit'], dtype=float)

    finite = np.isfinite(lam) & np.isfinite(residual)
    lam = lam[finite]
    residual = residual[finite]

    order = np.argsort(lam)
    lam = lam[order]
    residual = residual[order]

    wn = 1.0 / lam
    order_wn = np.argsort(wn)
    wn = wn[order_wn]
    residual = residual[order_wn]

    wn_uniform = np.linspace(wn[0], wn[-1], n_uniform)
    cs = spi.CubicSpline(wn, residual)
    residual_uniform = cs(wn_uniform)

    residual_uniform = residual_uniform - np.mean(residual_uniform)
    residual_windowed = residual_uniform * np.hanning(len(residual_uniform))

    dwn = wn_uniform[1] - wn_uniform[0]
    fft_vals = spf.fft(residual_windowed)
    fft_freq = spf.fftfreq(len(wn_uniform), d=dwn)

    pos = fft_freq > 0
    mod_freq = fft_freq[pos]
    mod_amp = np.abs(fft_vals[pos])

    valid = mod_freq >= min_mod_freq
    mod_freq_use = mod_freq[valid]
    mod_amp_use = mod_amp[valid]

    if len(mod_amp_use) < 10:
        print(f"\n[{label}] Not enough FFT bins after low-frequency cutoff.")
        return None

    prom = fft_peak_prominence_ratio * np.max(mod_amp_use)
    pk, _ = find_peaks(mod_amp_use, prominence=prom)

    if len(pk) == 0:
        pk = np.argsort(mod_amp_use)[-max_peaks:]

    pk = pk[np.argsort(mod_amp_use[pk])[::-1]]
    pk = pk[:max_peaks]

    peak_freqs = mod_freq_use[pk]
    peak_amps = mod_amp_use[pk]
    candidate_thicknesses = peak_freqs / (2.0 * refractive_index)

    order_pk = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[order_pk]
    peak_amps = peak_amps[order_pk]
    candidate_thicknesses = candidate_thicknesses[order_pk]

    median_amp = np.median(mod_amp_use)
    contrasts = peak_amps / median_amp if median_amp > 0 else np.full_like(peak_amps, np.inf)

    print(f"\n[{label}] --- Frequency analysis of benchmark residual ---")
    print(f"Number of candidate modulation peaks: {len(peak_freqs)}")
    print(f"Low-frequency cutoff: {min_mod_freq:.6e} cycles per (m^-1)")

    for i, (f, a, c, L) in enumerate(zip(peak_freqs, peak_amps, contrasts, candidate_thicknesses), start=1):
        print(f"Peak {i}:")
        print(f"  modulation frequency = {f:.6e} cycles per (m^-1)")
        print(f"  FFT amplitude        = {a:.6e}")
        print(f"  FFT contrast         = {c:.3f}")
        print(f"  candidate thickness  = {L:.6e} m")

    plt.figure(f'Benchmark residual FFT: {label}')
    plt.plot(mod_freq_use, mod_amp_use, label='FFT magnitude')
    plt.plot(peak_freqs, peak_amps, 'o', label='Detected FFT peaks')
    for f in peak_freqs:
        plt.axvline(f, linestyle='--', alpha=0.5)
    plt.xlabel('Modulation frequency [cycles per (m$^{-1}$)]')
    plt.ylabel('FFT magnitude')
    plt.title(f'FFT of benchmark residual: {label}')
    plt.grid()
    plt.legend()

    return {
        'peak_freqs': peak_freqs,
        'peak_amps': peak_amps,
        'contrasts': contrasts,
        'candidate_thicknesses': candidate_thicknesses
    }


# =========================================================
# 4. Remove bumps by detrending residual first, then smoothing
#    Plot only 4.3e-7 to 6.6e-7 m and integral-normalize
# =========================================================

def remove_bumps_by_residual_smoothing(
    benchmark_result,
    led_x,
    led_y,
    trend_window=151,
    bump_window=21,
    polyorder=3,
    plot_min=4.3e-7,
    plot_max=6.6e-7,
    label='white_light_4'
):
    """
    Step 1: residual = white - fitted benchmark background
    Step 2: remove slow drift from residual
    Step 3: smooth detrended residual to estimate bump component
    Step 4: subtract bump component from original spectrum
    Step 5: plot in chosen wavelength range with integral normalization
    """

    lam = np.array(benchmark_result['white_full_x'], dtype=float)
    white = np.array(benchmark_result['white_full_y'], dtype=float)
    background = np.array(benchmark_result['background_full'], dtype=float)

    finite = np.isfinite(lam) & np.isfinite(white) & np.isfinite(background)
    lam = lam[finite]
    white = white[finite]
    background = background[finite]

    order = np.argsort(lam)
    lam = lam[order]
    white = white[order]
    background = background[order]

    residual = white - background

    # Slow trend removal
    trend_window = int(trend_window)
    if trend_window >= len(residual):
        trend_window = len(residual) - 1
    if trend_window % 2 == 0:
        trend_window -= 1
    if trend_window < polyorder + 2:
        trend_window = polyorder + 3
        if trend_window % 2 == 0:
            trend_window += 1

    trend = savgol_filter(residual, trend_window, polyorder)
    residual_detrended = residual - trend

    # Estimate bumps from detrended residual
    bump_window = int(bump_window)
    if bump_window >= len(residual_detrended):
        bump_window = len(residual_detrended) - 1
    if bump_window % 2 == 0:
        bump_window -= 1
    if bump_window < polyorder + 2:
        bump_window = polyorder + 3
        if bump_window % 2 == 0:
            bump_window += 1
    
    print("trend_window used =", trend_window)
    print("bump_window used  =", bump_window)
    bumps = savgol_filter(residual_detrended, bump_window, polyorder)

    # Remove only bump component
    cleaned = white - bumps

    # LED interpolation
    led_x = np.array(led_x, dtype=float)
    led_y = np.array(led_y, dtype=float)

    finite_led = np.isfinite(led_x) & np.isfinite(led_y)
    led_x = led_x[finite_led]
    led_y = led_y[finite_led]

    order_led = np.argsort(led_x)
    led_x = led_x[order_led]
    led_y = led_y[order_led]

    interp_led = interp1d(
        led_x,
        led_y,
        bounds_error=False,
        fill_value=np.nan
    )
    led_interp = interp_led(lam)

    # Restrict to reliable plot range
    mask = (
        (lam >= plot_min) &
        (lam <= plot_max) &
        np.isfinite(white) &
        np.isfinite(cleaned) &
        np.isfinite(led_interp)
    )

    lam_plot = lam[mask]
    original_plot = white[mask]
    cleaned_plot = cleaned[mask]
    led_plot = led_interp[mask]
    residual_plot = residual[mask]
    trend_plot = trend[mask]
    detrended_plot = residual_detrended[mask]
    bumps_plot = bumps[mask]

    if len(lam_plot) < 10:
        raise ValueError("Not enough overlapping finite points in chosen plot range.")

    # Integral normalization
    orig_int = np.trapz(original_plot, lam_plot)
    clean_int = np.trapz(cleaned_plot, lam_plot)
    led_int = np.trapz(led_plot, lam_plot)

    original_norm = original_plot / orig_int
    cleaned_norm = cleaned_plot / clean_int
    led_norm = led_plot / led_int

    # Diagnostic plots
    plt.figure(f'Residual components: {label}')
    plt.plot(lam_plot, residual_plot, label='Original residual')
    plt.plot(lam_plot, trend_plot, label='Slow trend')
    plt.plot(lam_plot, detrended_plot, label='Detrended residual')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Residual')
    plt.title(f'Residual decomposition: {label}')
    plt.xlim(plot_min, plot_max)
    plt.grid()
    plt.legend()

    plt.figure(f'Estimated bump structure: {label}')
    plt.plot(lam_plot, bumps_plot, label='Extracted bump component')
    plt.axhline(0, linestyle='--')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Residual amplitude')
    plt.title('Estimated bump structure')
    plt.xlim(plot_min, plot_max)
    plt.grid()
    plt.legend()

    plt.figure(f'white_light_4 with bumps suppressed: {label}')
    #plt.plot(lam_plot, original_norm, label='Original white_light_4')
    plt.plot(lam_plot, cleaned_norm, label='Bumps suppressed')
    plt.plot(lam_plot, led_norm, label='White_LED_Lens')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Integral-normalized intensity')
    plt.title('white_light_4 with bumps suppressed')
    plt.xlim(plot_min, plot_max)
    plt.grid()
    plt.legend()

    return {
        'lambda_plot': lam_plot,
        'original_plot': original_norm,
        'cleaned_plot': cleaned_norm,
        'led_plot': led_norm,
        'bumps_plot': bumps_plot,
        'trend_plot': trend_plot,
        'detrended_plot': detrended_plot
    }



# =========================================================
# 5. Main
# =========================================================

white = data('white_light_4')
led = grating('White_LED_Lens')

white_x = np.array(white[1], dtype=float)
white_y = np.array(white[2], dtype=float)

led_x = np.array(led[1], dtype=float)
led_y = np.array(led[2], dtype=float)

print("LED x min/max:", np.nanmin(led_x), np.nanmax(led_x))

# Original comparison over requested range, integral-normalized
orig_mask_white = (white_x >= 4.3e-7) & (white_x <= 6.6e-7)
orig_mask_led = (led_x >= 4.3e-7) & (led_x <= 6.6e-7)

white_x_plot = white_x[orig_mask_white]
white_y_plot = white_y[orig_mask_white]

led_x_plot = led_x[orig_mask_led]
led_y_plot = led_y[orig_mask_led]

white_y_plot_norm = white_y_plot / np.trapz(white_y_plot, white_x_plot)
led_y_plot_norm = led_y_plot / np.trapz(led_y_plot, led_x_plot)

plt.figure('Original comparison')
plt.plot(white_x_plot, white_y_plot_norm, label='white_light_4')
plt.plot(led_x_plot, led_y_plot_norm, label='White_LED_Lens')
plt.xlabel('Wavelength (m)')
plt.ylabel('Integral-normalized intensity')
plt.title('Original white_light_4 vs White_LED_Lens')
plt.xlim(4.3e-7, 6.6e-7)
plt.grid()
plt.legend()

# Benchmark fit
benchmark_result = fit_led_benchmark_background(
    white_x,
    white_y,
    led_x,
    led_y,
    lam_min=4.3e-7,
    lam_max=6.6e-7,
    label='white_light_4'
)

# Optional: frequency analysis of benchmark residual
freq_result = frequency_analysis_from_benchmark_residual(
    benchmark_result,
    n_uniform=2000,
    min_mod_freq=2.0e-5,
    fft_peak_prominence_ratio=0.05,
    max_peaks=5,
    refractive_index=1.0,
    label='white_light_4'
)

# Bump suppression by detrending + smoothing
clean_result = remove_bumps_by_residual_smoothing(
    benchmark_result,
    led_x,
    led_y,
    trend_window=15,
    bump_window=7,
    polyorder=3,
    plot_min=4.3e-7,
    plot_max=6.6e-7,
    label='white_light_4'
)

plt.show()
