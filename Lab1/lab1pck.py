import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.signal.windows import blackmanharris


def load_data(datafile, n_block):
    ''' Loads in SDR data from .npz and removes the solid current constant (DC) from the data, returns array.
    datafile: array of data from SDR
    n_block: integer value from n-blocks of data sampled, first two blocks are subject to buffer error'''
    loaded = np.load(datafile) 
    data = loaded["data"]
    fs   = loaded["fs"]
    signal = data[n_block].astype(float)
    signal = signal - np.mean(signal)
    return signal, fs


def apply_bh_window(signal):
    '''Applies a Blackman-Harris window to the input data array. 
    signal: original signal data, array'''
    window = blackmanharris(len(signal))
    return signal * window


def idealsine_plot(signal, fs, input_freq, xlim=None):
    '''Plots experimental data as fitted to an ideal sine wave by using the amplitude and phase calculated from input data.
    signal: array of experimental data
    fs: sample frequency, integer value in hertz
    input_freq: input frequency, integer value in hertz
    xlim: specified xlimit, ordered pair'''
    N = len(signal)
    t = np.arange(N) / fs
    w = 2 * np.pi * input_freq
    
    #least squares fit for phase (not amplitude)
    X = np.column_stack([np.sin(w*t), np.cos(w*t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
    C, D = coeffs
    
    # Amplitude from data (not from fit)
    A = (np.max(signal) - np.min(signal)) / 2
    

    phi = np.arctan2(D, C)

    ideal_sine = A * np.sin(w * t + phi)

    plt.figure(figsize=(6,4))
    plt.plot(ideal_sine, color='tab:cyan', label=f"Fit Ideal sine, A:\n{A:.2f}, phi:\n{phi:.2f}", linewidth=2.5)
    plt.plot(signal, color='darkslategrey', label="Experimental Data", linestyle='--', marker='o')
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper left')
    plt.grid()
    
    if xlim is not None: 
        plt.xlim(xlim)
        
    plt.show()
    
    return ideal_sine, A, phi


def compute_fft(signal, fs, window=None):
    '''Computes Fast Fourier Transform of the data.
    signal: original data, array
    fs: sample frequency, integer
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)'''

    if window is not None:
        signal = window(signal)

    N = len(signal)
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)

    mask = freqs > 0

    return freqs[mask], fft[mask]


def voltagespectra_plot(signal, fs, input_freq=None, window=None, xlim=None):
    '''Plots voltage spectra by applying a FFT to the input data.
    signal: array of input data
    fs: sample frequency, integer value in hertz
    input_freq: input frequency, integer value in hertz
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)
    xlim: ordered pair to specify axes limits'''
    
    freqs, fft = compute_fft(signal, fs, window)
    
    plt.figure(figsize=(6,4))
    plt.plot(freqs/1e3, np.abs(fft), color='navy')
    
    if input_freq:
        plt.axvline(input_freq/1e3, linestyle="--", label="Input freq", color='magenta')

    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Voltage Spectrum")
    plt.grid()
    plt.legend()
    
    if xlim: 
        plt.xlim(xlim)
        
    plt.show()
    
    
def powerspectra_plot(signal, fs, input_freq=None, window=None, xlim=None):
    '''Plots power spectra by applying absolute value and square root to the FFT of the signal provided. 
    signal: array of input data
    fs: sample frequency, integer value in hertz
    input_freq: input frequency, integer value in hertz
    input_freq: input frequency, integer value in hertz
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)
    xlim: ordered pair to specify axes limits'''
    freqs, fft = compute_fft(signal, fs, window)
    power = np.abs(fft)**2

    plt.figure(figsize=(6,4))
    plt.semilogy(freqs/1e3, power, color='mediumvioletred')
    
    if input_freq:
        plt.axvline(input_freq/1e3, linestyle="--", label="Input freq", color='deepskyblue')
    
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (arbitrary units)')
    plt.legend()
    plt.grid()
    
    if xlim: 
        plt.xlim(xlim)
        
    plt.show()
    

def compare_power_spectra(signal_clean, signal_noisy, fs_c, fs_n, xlim=None):
    '''Plot multiple power spectras to compare them.
    signal_clean: array
    signal_noisy: array
    fs: sampling rate, integer (one for each in case they are different values)'''

    freqs_nobh, fft_nobh = compute_fft(signal_clean, fs_c, window=None)
    freqs_bh, fft_bh = compute_fft(signal_clean, fs_c, window=apply_bh_window)
    freqs_noisy, fft_noisy = compute_fft(signal_noisy, fs_n, window=None)

    power_nobh = np.abs(fft_nobh)**2
    power_bh = np.abs(fft_bh)**2
    power_noisy = np.abs(fft_noisy)**2

    plt.figure(figsize=(6,4))
    plt.semilogy(freqs_nobh/1e3, power_nobh, label="Clean Signal, Spectral Leakage", color='mediumaquamarine', alpha=1.0)
    plt.semilogy(freqs_bh/1e3, power_bh, label="Clean Signal, BH Window", color='darkviolet', alpha=0.5)
    plt.semilogy(freqs_noisy/1e3, power_noisy, label="Noisy Signal", color='royalblue', alpha=0.3)
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.grid()
    plt.legend()

    if xlim: 
        plt.xlim(xlim)
        
    plt.show()

    
    
def gaussianfit_noise(noise, bins=100, xlim=None, ylim=None):
    '''Plots the noise in a histogram and fits a gaussian on top to show trend. To prove validity of fit, plots residuals of histogram. 
    noise: array of input data
    bins: number of bins in histogram, integer value
    xlim: ordered pair for x axis limits
    ylim: ordered pair for y axis limits'''
    def gaussian(x, A, mu, sigma): 
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    counts, edges = np.histogram(noise, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    initial_guesses = [max(counts), np.mean(noise), np.std(noise)]
    popt, pcov = curve_fit(gaussian, centers, counts, p0=initial_guesses, maxfev=10000)
    fit_amplitude, fit_mean, fit_std_dev = popt
    
    x_values = np.linspace(min(centers), max(centers), len(counts))
    y_fit = gaussian(x_values, *popt)
    
    residuals = y_fit - counts
    
    plt.figure(figsize=(6,4))
    plt.hist(noise, bins=bins, label='Noise', alpha=0.5, color='mediumslateblue', edgecolor='slateblue')
    plt.plot(x_values, y_fit, color='blue', label=f'Gaussian Fit:\n mean={fit_mean:.2f}, std={fit_std_dev:.2f}')
    plt.plot(x_values, residuals, label='Residuals', color='navy', marker='o', linestyle='none')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()

    if xlim:
        plt.xlim(xlim)
        
    if ylim: 
        plt.ylim(ylim)
            
    plt.show()
    
    #return popt


def nyquist_plot(signal, fs, true_freq=None, window=None):
    '''Plots the Nyquist frequency and shows aliasing if present
    signal: input data, array
    fs: sample frequency, integer
    true_freq: input frequeny, integer in hertz
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)
    '''

    freqs, fft = compute_fft(signal, fs, window)
    power = np.abs(fft)**2

    nyquist = fs/2

    plt.figure(figsize=(6,4))
    plt.semilogy(freqs/1e3, power, label="Spectrum", color='mediumturquoise')

    plt.axvline(nyquist/1e3, linestyle="--", color="black",
                label="Nyquist limit")

    if true_freq:
        alias = abs(true_freq - round(true_freq/fs)*fs)

        plt.axvline(true_freq/1e3, linestyle=":", label="True freq", color='orangered')
        plt.axvline(alias/1e3, linestyle=":", label="Alias freq", color='forestgreen')

    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid()
    plt.show()

    
def recover_frequency(signal, fs, window=None, fmin=10e3):
    '''Recovers the correct input frequency from the original data AS LONG AS there is no aliasing.
    signal: original data, array
    fs: sample frequency, integer
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)
    fmin: input to ignore peaks in the data that are not from aliasing, integer value'''

    freqs, fft = compute_fft(signal, fs, window)
    power = np.abs(fft)**2

    # Ignore low-frequency region
    valid = freqs > fmin

    freqs = freqs[valid]
    power = power[valid]

    idx = np.argmax(power)
    return freqs[idx]


def reconstruct_true_frequency(input_freq, fs):
    '''Given observed frequency after sampling, compute possible true frequencies.
    input_freq: input frequency, integer
    fs: sample frequency, integer
    '''
    nyquist = fs / 2

    result = {
        "observed_frequency": input_freq,
        "nyquist_frequency": nyquist,
        "aliasing_possible": False,
        "possible_true_frequencies": [input_freq]
    }

    if input_freq > nyquist:
        result["aliasing_possible"] = True
        result["possible_true_frequencies"].append(abs(fs - input_freq))

    return result

def predict_possible_inputs(observed_freq, fs, max_multiple=3):
    '''Given an observed (possibly aliased) frequency, return possible true input frequencies.
    observed_freq: frequency recovered from FFT
    fs: sampling frequency
    max_multiple: how many multiples of fs to consider
    '''

    possible = []

    for n in range(max_multiple + 1):
        f1 = n * fs + observed_freq
        f2 = abs(n * fs - observed_freq)

        if f1 >= 0:
            possible.append(f1)

        if f2 >= 0:
            possible.append(f2)

    possible = sorted(list(set(possible)))

    return {
        "observed_frequency": observed_freq,
        "sampling_frequency": fs,
        "possible_true_frequencies": possible
    }



def detect_alias(signal, fs, window=None, fmin=10e3, true_input=None):
    '''Determines if aliasing is present or not given the input data and sample frequency, basically a check if Nyquist criterion is violated.
    signal: original data, array
    fs: sample frequency, integer
    window: function that acts on the signal to add a window (e.g. Blackman-Harris)
    fmin: input to ignore peaks in the data that are not from aliasing, integer value
    true_input: actual input frequency, for the case that frequency is above nyquist'''
    
    obs_freq = recover_frequency(signal, fs, window, fmin)
    possible_inputs = predict_possible_inputs(obs_freq, fs)
    nyquist = fs / 2

    result = {
        "observed_frequency": obs_freq,
        "nyquist_frequency": nyquist,
        "aliasing_detected": None,
        "possible_true_frequencies": possible_inputs["possible_true_frequencies"],
        "mode": None
    }
    
    if true_input is not None:
        result["mode"] = "known_input"

        if true_input > nyquist:
            result["aliasing_detected"] = True
        else:
            result["aliasing_detected"] = False

        result["true_input_frequency"] = true_input

    else:
        result["mode"] = "unknown_input"

        result["aliasing_detected"] = "Indeterminate from sampled data alone"

    return result


