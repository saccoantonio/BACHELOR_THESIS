import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
from tqdm import tqdm # type: ignore
import gc
import glob
import os
import warnings

plt.style.use(["science", "nature"])
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 40,
        "legend.title_fontsize": 12,
        "figure.figsize": (8.0, 6.2),
        "axes.linewidth": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.alpha": 0.25,
    }
)


# Simple methods


def autocorrelation_fft(x, stride=1):
    """
    Compute the autocorrelation function (ACF) using FFT,
    with optional undersampling (stride).

    Parameters
    ----------
    x : array-like
        Input signal (e.g., heat flux time series).
    stride : int, optional
        Undersampling factor. If stride > 1, only every 'stride'-th
        element of x is used to compute the ACF. Default is 1 (no undersampling).

    Returns
    -------
    acf : np.ndarray
        Normalized autocorrelation function (acf[0] = 1).
    """

    # --- Optional undersampling ---
    if stride > 1:
        x = x[::stride]

    # Remove mean
    x = x - np.mean(x)
    N = len(x)

    # Compute the Fast Fourier Transform (FFT) of x, 
    # padding to length 2N to reduce circular convolution (aliasing) effects
    fft_x = np.fft.fft(x, n=2*N)

    # Compute the inverse FFT of the power spectrum (fft_x * conjugate(fft_x)),
    # which gives the autocorrelation function (real part only, first N elements)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:N].real

    # Normalize the autocorrelation so that acf[0] = 1,
    # dividing by the variance and the decreasing number of overlapping points
    acf /= (x.var() * (N - np.arange(N)))

    return acf


def plot_acf(t, ac, acfit):
    """Plot the autocorrelation functions in 3 directions (or total vs fitted)."""
    
    # Create a new figure with a specific size (13x5 inches)
    plt.figure(figsize=(13, 5))
    
    # Plot the total autocorrelation function versus time
    plt.plot(t, ac, label="total acf")

    # Plot the fitted autocorrelation function versus time
    plt.plot(t, acfit,  label="fit acf")
    
    # Draw a horizontal dashed line at y=0 for visual reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set the y-axis label using LaTeX syntax for the correlation formula
    plt.ylabel("HFACF", fontsize=17)
    
    # Set the x-axis label to indicate correlation time in picoseconds
    plt.xlabel("correlation length (ps)", fontsize=12)
    
    # Use a logarithmic scale for the x-axis to better show decaying behavior
    plt.xscale('log')
    
    # Add a legend in the lower left corner to distinguish the curves
    plt.legend(loc="lower left")
    plt.grid(True, which='both', ls='--', alpha=0.4)

    # Display the plot
    plt.show()


def ACF_fit(x, A, B, C, n, m):
    """Double exponential fit for the autocorrelation function (ACF)."""
    
    # Avoid division by zero or extremely small denominators for parameter n
    # If |n| is smaller than 1e-12, replace it with ±1e-12 preserving the sign
    if np.abs(n) < 1e-12: 
        n = np.sign(n) * 1e-12
    
    # Same safeguard for parameter m
    if np.abs(m) < 1e-12: 
        m = np.sign(m) * 1e-12
    
    # Compute the exponent arguments for the two exponential terms
    # np.clip limits values between -700 and 700 to avoid numerical overflow
    z1 = np.clip(-x/n, -700, 700)
    z2 = np.clip(-x/m, -700, 700)
    
    # Return the sum of two exponential decays plus a constant offset C = (1 - A - B) 
    # such that Impone f(0) = A + B + (1 - A - B) = 1
    # A and B are amplitudes, n and m are decay constants 
    return A * np.exp(z1) + B * np.exp(z2) + C #(1 - A - B)


def fit_acf(x, ac):
    """Fit the autocorrelation function (ACF) using a double exponential model."""
    
    # Perform a nonlinear least-squares fit of the ACF data (ac vs x)
    # using the ACF_fit function defined earlier as the model.
    # 'pars' contains the best-fit parameters [A, B, C, n, m]
    # 'cov' is the covariance matrix of the parameters (not used here).
    # maxfev=10000 increases the maximum number of function evaluations 
    # to help the optimizer converge for complex fits.
    pars, cov = curve_fit(ACF_fit, x, ac, maxfev=10000)
    
    # Return only the fitted parameters
    return pars


def compute_F_E(ac, J, x, delta=1000, prominence=0.0007, n_points=100000): 
    
    """Compute the fluctuation function F(t) and the averaged autocorrelation E(t) 
    from the autocorrelation function (ac) and time series J.

    Parameters:
        ac : array-like
            Autocorrelation function values.
        J : array-like
            Original time series data (e.g., flux).
        x : array-like or None
            Time array corresponding to ac and J.
        delta : int
            Window length for local averaging/variance.
        prominence : float
            Minimum prominence to detect peaks in F.
        n_points : int
            Number of points to process for F and E.

    Returns:
        F : array
            Fluctuation function F(t).
        E : array
            Local mean of the autocorrelation function E(t).
        first_peak_x : float or None
            x-coordinate of the first significant peak in F(t).
        first_zero_x : float or None
            x-coordinate of the first zero crossing of E(t)."""
    
    # Initialize E array
    E = np.zeros(len(ac))

    # Initialize F array
    F = np.zeros(len(ac))

    # Scale autocorrelation by variance of J
    cor = ac * J.var()

    # Compute local mean of the autocorrelation function E(t)
    # Compute fluctuation function F(t): ratio of standard deviation to mean in the window
    for j in range(min(len(ac) - delta, n_points)):
        E[j] = np.mean(ac[j:j+delta])
        F[j] = np.sqrt(np.var(cor[j:j+delta]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        F /= abs(E * J.var())

    # --- Initialize output variables for first peak and zero crossing ---
    first_peak_x = None
    first_zero_x = None
    
    if x is not None:
        # ---- FIND FIRST PEAK OF F(t) ----
        # Normalize F for peak detection
        F_norm = F[:n_points] / np.max(F[:n_points])
        
        # Use scipy.signal.find_peaks to locate peaks above given prominence
        peaks, _ = find_peaks(F_norm, prominence=prominence)
        
        if len(peaks) > 0:
            # Take the first significant peak
            first_peak_idx = peaks[0]
            first_peak_x = x[first_peak_idx]
        else:
            print("No significant peak found in F(t).")
        
        # ---- FIND FIRST ZERO CROSSING OF E(t) ----
        # Detect sign changes (zero crossings) in E
        sign_change_idx = np.where(np.diff(np.sign(E[:n_points])))[0]
        
        if len(sign_change_idx) > 0:
            # Take the first zero crossing and interpolate linearly
            first_zero_idx = sign_change_idx[0]
            x0, x1 = x[first_zero_idx], x[first_zero_idx+1]
            y0, y1 = E[first_zero_idx], E[first_zero_idx+1]
            first_zero_x = x0 - y0*(x1 - x0)/(y1 - y0)  # linear interpolation
    
    # Return fluctuation function, local mean, first peak, and first zero
    return F, E, first_peak_x, first_zero_x


def plot_F_E(x, F, first_peak_x, E, first_zero_x, n_plot=100000, dt=0.001):
    """
    Plot the fluctuation function F(t) and the averaged autocorrelation E(t)
    with markers for the first significant peak and first zero crossing.
    
    Parameters:
        x : array-like
            Time or correlation length array.
        F : array-like
            Fluctuation function F(t).
        first_peak_x : float
            x-coordinate of the first significant peak in F(t).
        E : array-like
            Local mean of autocorrelation function E(t).
        first_zero_x : float
            x-coordinate of the first zero crossing of E(t).
        n_plot : int
            Number of points to plot.
    """

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    # --- Plot F(t) on the left subplot ---
    ax[0].plot(x[:n_plot], F[:n_plot])
    ax[0].set_ylabel("$F(t)$", fontsize=17)
    ax[0].set_xlabel("correlation length (ps)", fontsize=12)
    
    # Use logarithmic scale for both axes to show wide range
    # ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    # Set y-axis limits to avoid extreme values
    ax[0].set_ylim([0.1, 1e6])
    
    # Mark the first significant peak with a red dot
    ax[0].plot(first_peak_x, F[int(first_peak_x/dt)], "ro", label=f"Significant peak at t={first_peak_x:.3f}ps")
    ax[0].annotate('', xy=(first_peak_x, F[int(first_peak_x/dt)]), xytext=(first_peak_x-5, F[int(first_peak_x/dt)]*10),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'), color='red')

    # Add legend
    ax[0].legend(loc="upper right")
    ax[0].grid(True, which='both', ls='--', alpha=0.4)

    
    # --- Plot E(t) on the right subplot ---
    ax[1].plot(x[:n_plot], E[:n_plot])
    
    # Draw horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax[1].set_ylabel("$E(t)$", fontsize=17)
    ax[1].set_xlabel("correlation length (ps)", fontsize=12)
        
    # Mark the first zero crossing with a red dot
    ax[1].plot(first_zero_x, 0, "ro", label=f"First zero at t={first_zero_x:.3f}ps")  
    ax[1].annotate('', xy=(first_zero_x, 0), xytext=(first_zero_x, -0.06),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'), color='red')
    
    # Add legend
    ax[1].legend(loc="upper right")
    ax[1].grid(True, which='both', ls='--', alpha=0.4)
    
    # Adjust subplot layout for better spacing
    plt.tight_layout()
    
    # Display the figure
    plt.show()


def integrate_acf(ac, dt=0.001):
    """
    Numerically integrate the autocorrelation function ACF(t) using the trapezoidal rule.
    Vectorized version for high efficiency (O(N) instead of O(N²)).
    
    Parameters:
        ac : array-like
            Autocorrelation function values.
        dt : float
            Time step between consecutive ACF points.
    
    Returns:
        kk : array
            Cumulative integral of ACF(t) at each time point.
    """
    
    # Compute the area between consecutive points using the trapezoidal rule:
    # 0.5 * dt * (f_i + f_{i+1}) for each interval
    k = 0.5 * dt * (ac[1:] + ac[:-1])
    
    # Compute the cumulative sum to get the integral at each point
    # Prepend 0.0 to indicate the integral at the first point is zero
    kk = np.concatenate(([0.0], np.cumsum(k)))
    
    # Return the cumulative integral
    return kk


def compute_kappa(kk, J, V, T):
    """
    Compute the thermal conductivity κ from the integrated autocorrelation function.
    
    Parameters:
        kk : array-like
            Cumulative integral of the heat flux autocorrelation (from integrate_acf).
        J : array-like
            Original heat flux time series.
        V : float
            Volume of the system.
        T : float
            Temperature of the system (in Kelvin).
    
    Returns:
        kappa : array
            Thermal conductivity in W/mK (SI units).
    """
    
    # Scale factor according to the Green-Kubo formula:
    # kappa = (1 / (V * T^2)) * integral(<J(0) J(t)>) * conversion factor
    # 8.61673324e-5 is the Boltzmann constant in eV/K
    # Multiply by variance of J to scale the normalized autocorrelation
    # V is in the denominator beacuse J is not normalized by V
    scale = 1 / (V * T**2) / (8.61673324*10**(-5)) * J.var()
    
    # Conversion factor from eV/(ps·Å²·K) to W/mK
    metal2SI = 1.602176634*10**3
    
    # Return the thermal conductivity in SI units
    return kk * scale * metal2SI


def compute_kappa_FA_FD(x, kap, first_peak_x, first_zero_x, timestep):
    """
    Compute thermal conductivity using two approximate methods:
    FA (First Avalanche) and FD (First Dip).
    
    Parameters:
        x : array-like
            Time or correlation length array.
        kap : array-like
            Thermal conductivity as a function of time.
        first_peak_x : float or None
            Time of first significant peak in F(t).
        first_zero_x : float or None
            Time of first zero crossing in E(t).
        timestep : float
            Time step between consecutive points.
    
    Returns:
        tau_FA : float or None
            Time used for FA method.
        kappa_FA : float or None
            Thermal conductivity from FA method.
        tau_FD : float
            Time used for FD method.
        kappa_FD : float
            Thermal conductivity from FD method.
    """
    
    # --- FA (First Avalanche) Method ---
    tau_FA = None
    kappa_FA = None
    if first_peak_x is not None:
        # Set the time of the first peak as tau_FA
        tau_FA = first_peak_x
        # Get the corresponding thermal conductivity value from kap array
        kappa_FA = kap[int(first_peak_x / timestep)]
    
    # --- FD (First Dip) Method ---
    if first_zero_x is not None:
        # Set the time of first zero crossing as tau_FD
        tau_FD = first_zero_x
        # Get the corresponding thermal conductivity value
        kappa_FD = kap[int(first_zero_x / timestep)]
    else:
        # Fallback: if no zero crossing, take the maximum of the kappa(t) curve
        fd_idx = np.argmax(kap)
        tau_FD = x[fd_idx]
        kappa_FD = kap[fd_idx]
    
    # Return both FA and FD times and thermal conductivities
    return tau_FA, kappa_FA, tau_FD, kappa_FD


def plot_kappa_FA_FD(x, kap, fitkap, tau_FA, kappa_FA, tau_FD, kappa_FD, n_plot=500000):
    """
    Plot the thermal conductivity κ(t) from raw and fitted integration,
    marking FA (First Avalanche) and FD (First Dip) methods.
    
    Parameters:
        x : array-like
            Time or correlation length array.
        kap : array-like
            Thermal conductivity from raw integration.
        fitkap : array-like
            Thermal conductivity from double-exponential fit integration.
        tau_FA, kappa_FA : float
            Time and κ value for FA method.
        tau_FD, kappa_FD : float
            Time and κ value for FD method.
        n_plot : int
            Number of points to plot.
    """
    
    # Create a figure with specific size
    plt.figure(figsize=(13, 5))
    
    # Plot raw thermal conductivity (from cumulative integration)
    plt.plot(x[:n_plot], kap[:n_plot], label="raw integration")
    
    # Plot thermal conductivity obtained from double exponential fit
    plt.plot(x[:n_plot], fitkap[:n_plot], label="double exponential fit integration")

    # Draw horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Mark FA method point with red circle and label
    plt.plot(tau_FA, kappa_FA, "ro", 
             label=f"FA: τc={tau_FA:.2f} ps, κ={kappa_FA:.3f} W/mK")
    
    # Mark FD method point with green square and label
    plt.plot(tau_FD, kappa_FD, "gs", 
             label=f"FD: τc={tau_FD:.2f} ps, κ={kappa_FD:.3f} W/mK")
    
    # Set y-axis label with LaTeX formatting
    plt.ylabel(r"$\kappa \; \frac{J}{m\cdot s \cdot K}$", fontsize=16)
    
    # Set x-axis label
    plt.xlabel("correlation length (ps)", fontsize=12)
    
    # Add legend to distinguish curves and points
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    
    # Display the plot
    plt.show()
    
    # Print FA and FD method results in the console
    print(f"Metodo FA → τc = {tau_FA:.2f} ps, κ = {kappa_FA:.3f} W/mK")
    print(f"Metodo FD → τc = {tau_FD:.2f} ps, κ = {kappa_FD:.3f} W/mK")


def compute_kappa_Chen(pars, J, V, T, tau_FA):
    """
    Compute thermal conductivity using Chen's method with double-exponential fit parameters.
    
    Parameters:
        pars : list or array-like
            Fit parameters from double exponential fit: [A1, A2, Y0, tau1, tau2]
        J : array-like
            Heat flux time series.
        V : float
            Volume of the system.
        T : float
            Temperature (Kelvin).
        tau_FA : float
            Correlation time from FA (First Avalanche) method.
    
    Returns:
        kappa_C : float
            Thermal conductivity computed from Chen's method without Y0 contribution.
        kappa_F : float
            Thermal conductivity including Y0 * tau_FA contribution.
    """
    
    # Unpack double-exponential fit parameters
    A1, A2, Y0, tau1, tau2 = pars
    # Y0 = 1 - A1 - A2 

    # Scale factor according to Green-Kubo formula
    # 8.61673324e-5 is Boltzmann constant in eV/K
    # Multiply by variance of J to scale normalized ACF
    scale = 1 / (V * T**2) / (8.61673324*10**(-5)) * J.var()
    
    # Conversion factor from eV/(ps·Å²·K) to W/mK
    metal2SI = 1.602176634*10**3
    
    # Compute kappa using only A1, A2, tau1, tau2
    kappa_C = scale * (A1 * tau1 + A2 * tau2) * metal2SI
    
    # Compute kappa including Y0 contribution times tau_FA
    kappa_F = scale * (A1 * tau1 + A2 * tau2 + Y0 * tau_FA) * metal2SI
    
    # Return both conductivity values
    return kappa_C, kappa_F


# "Advanced" methods


def plot_data(filenames, path, dir, labels, timesteps,
              segments=None, max_points=50000):
    """
    Plot thermodynamic data from multiple LAMMPS output files, optionally
    concatenating different simulation segments with proper time offsets.

    The function reads LAMMPS-style thermo output files, downsamples the data
    if necessary, concatenates time axes across multiple runs, and produces
    a multi-panel plot of selected thermodynamic quantities. Statistical
    analysis (mean, standard deviation, error, and drift) is performed only
    on the last file and written to a log file.

    Parameters
    ----------
    filenames : list of str
        Names of the thermo output files to be processed.
    path : str
        Directory containing the input files.
    dir : str
        Output directory where plots and logs will be saved.
    labels : list of str
        Labels for each file, used in the plot legend.
    timesteps : list of float
        Timestep size for each file (in femtoseconds).
    segments : list of int, optional
        Segment identifiers for each file. Files belonging to the same
        segment are concatenated without resetting the time offset.
        If None, each file is treated as a separate segment.
    max_points : int, optional
        Maximum number of points to plot per file after downsampling.

    Returns
    -------
    None
        The function saves a PNG figure and a text log file to disk.
    """

    # Thermodynamic quantities to be plotted
    cols_to_plot = ["v_T", "v_PE", "v_Etot", "v_P"]

    # If no segment information is provided, treat each file as a new segment
    if segments is None:
        segments = list(range(1, len(filenames) + 1))

    # Create output directory if it does not exist
    os.makedirs(dir, exist_ok=True)

    # Create a grid of subplots (2 columns)
    fig, axes = plt.subplots(int(len(cols_to_plot) / 2), 2, figsize=(8, 5))
    axes = axes.flatten()

    
    # Compute time offsets to concatenate multiple runs seamlessly
    offsets = [0.0]

    for k in range(1, len(filenames)):
        # Read previous file to determine its total simulated duration
        prev_df = pd.read_csv(
            os.path.join(path, filenames[k - 1]),
            sep=r"\s+", comment="#", header=None
        )

        # Duration = last timestep index * timestep size
        duration = prev_df.iloc[-1, 0] * timesteps[k - 1]

        # Increase offset only if the segment changes
        if segments[k] == segments[k - 1]:
            offsets.append(offsets[-1])
        else:
            offsets.append(offsets[-1] + duration)

    # Open log file for statistical analysis
    log_path = os.path.join(dir, "therm.txt")
    with open(log_path, "w") as log:

        # Loop over all input files
        for i_file, (fname, label, timestep) in enumerate(
            zip(filenames, labels, timesteps)
        ):

            fullpath = os.path.join(path, fname)

            
            # Read column names from the second line of the file
            with open(fullpath) as f:
                f.readline()          # Skip description line
                header_line = f.readline()  # Column names line

            # Remove the leading '#' and store column names
            columns = header_line.split()[1:]

            # Read numerical data, skipping the first two lines
            df = pd.read_csv(
                fullpath,
                sep=r"\s+",
                skiprows=2,
                names=columns
            )

            
            # Downsampling to limit the number of plotted points
            if len(df) > max_points:
                stride = len(df) // max_points
                df = df.iloc[::stride]

            # Construct time axis (converted to ns in the plot)
            x = df["TimeStep"].values * timestep + offsets[i_file]

            # Loop over all quantities to be plotted
            for j, col in enumerate(cols_to_plot):

                # Skip quantities not present in the file
                if col not in df.columns:
                    continue

                y = df[col].values

                
                # Statistical analysis ONLY on the last file
                if i_file == len(filenames) - 1:

                    mean_val = np.mean(y)
                    std_val = np.std(y, ddof=1)
                    err_val = std_val / np.sqrt(len(y))

                    # Linear regression to estimate drift
                    slope, intercept, *_ = linregress(x, y)
                    duration = x[-1] - x[0]
                    drift_tot = slope * duration

                    # Special treatment for total energy
                    if col == "v_Etot":
                        drift_rel = drift_tot / mean_val
                        drift_str = (
                            "negligible" if abs(drift_rel) < 1e-5
                            else "significant"
                        )
                        print(
                            f"{fname:<15s} | {col:5s} -> "
                            f"mean={mean_val:.4f}, err={err_val:.4e}, "
                            f"drift_rel={drift_rel:.4e}, "
                            f"treshold={1e-5:.4e} ({drift_str})",
                            file=log
                        )
                    else:
                        drift_str = (
                            "negligible" if abs(drift_tot) < std_val
                            else "significant"
                        )
                        print(
                            f"{fname:<15s} | {col:6s} -> "
                            f"mean={mean_val:.4f}, err={err_val:.4e}, "
                            f"std={std_val:.4e}, drift={drift_tot:.4e} "
                            f"({drift_str})",
                            file=log
                        )

                    # Set y-limits based on fluctuations
                    axes[j].set_ylim(
                        mean_val - 8 * std_val,
                        mean_val + 8 * std_val
                    )

                
                # Plot data
                axes[j].plot(x / 1000, y, label=label)
                axes[j].set_xlabel("Time [ns]")
                axes[j].set_ylabel(col.replace("v_", ""))
                axes[j].grid(ls="--", alpha=0.4)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "therm.png"), dpi=300)

    print(f"Saved plot in {dir}/therm.png and log in therm.txt")


def analyze_kappa(file_path, dt=0.001, direction='x', window_size=100000, step_size=100000,
                  start_from=0, volume=None, plot_acf_flag=False, plot_EF_flag=False, plot_kappa_flag=False,
                  out_dir="prod"):

    """
    Analyze thermal conductivity using the Green-Kubo method with a sliding
    window approach.

    The function processes a LAMMPS-style heat flux file and computes the
    thermal conductivity along a selected spatial direction using direct
    ACF integration, fitted ACF, FA/FD characteristic times, and the
    Chen et al. method.

    Results are saved for each sliding window in compressed .npz files.

    Parameters
    ----------
    file_path : str
        Path to the input file containing heat flux and thermodynamic data.
    dt : float, optional
        Time step between consecutive data points.
    direction : {'x', 'y', 'z'}, optional
        Spatial direction for heat flux analysis.
    window_size : int, optional
        Number of time steps per sliding window.
    step_size : int, optional
        Step between consecutive windows.
    start_from : int, optional
        Index of the first data point used for windowing.
    volume : float or None, optional
        System volume. If None, the volume is read from the file.
    plot_acf_flag : bool, optional
        Plot the autocorrelation function and its fit.
    plot_EF_flag : bool, optional
        Plot fluctuation F(t) and mean HFACF E(t).
    plot_kappa_flag : bool, optional
        Plot thermal conoductivity curves and FA/FD markers.
    out_dir : str, optional
        Output directory.

    Returns
    -------
    None
    """
        
    print(f"Analyzing thermal conductivity in direction '{direction}'")
    
    # Create output directories if they do not exist
    dir_npz = os.path.join(out_dir, direction, "npz")
    os.makedirs(dir_npz, exist_ok=True)
    
    # Count total lines in file to determine number of points
    n_lines = sum(1 for _ in open(file_path))
    n_points = n_lines

    # Leggi la seconda riga del file (quella con i nomi delle colonne)
    with open(file_path) as f:
        f.readline()  # salta la prima riga (descrizione)
        header_line = f.readline()  # seconda riga con nomi

    # Trasforma in lista e rimuovi il primo elemento '#'
    columns = header_line.split()[1:]  # prende tutti tranne il primo

    # --- Sliding window loop ---
    for start in tqdm(range(start_from, n_points - window_size + 1, step_size), desc="Sliding windows"):
        end = start + window_size
        
        # Load only the current block from file using pandas
        df = pd.read_csv(
            file_path, sep=r"\s+", comment="#", header=None,
            names=columns, skiprows=start + 2, nrows=window_size
        )
        
        df = df.dropna()
        if len(df) < window_size:
            # Skip incomplete blocks
            print(f"Block {start}-{end} incomplete, skipping.")
            continue
        
        # --- Extract main quantities ---
        t = np.arange(window_size) * dt  # Time axis
        J = df[f"v_J{direction}"].values   # Heat flux in selected direction
        T = df["v_T"].mean()               # Average temperature
        V=0
        if volume is not None:
            V=volume
        else:
            V = df["v_V"].mean()  

        
        # --- Compute autocorrelation and integrate ---
        ac = autocorrelation_fft(J)      # Compute ACF
        kk = integrate_acf(ac, dt)           # Integrate ACF
        kap = compute_kappa(kk, J, V, T) # Compute thermal conductivity
        
        # --- Fit ACF with double exponential ---
        pars = fit_acf(t, ac)
        ac_fit = ACF_fit(t, *pars)
        if plot_acf_flag:
            plot_acf(t, ac, ac_fit)
        
        # Compute integrated kappa from fitted ACF
        kk_fit = integrate_acf(ac_fit, dt)
        kap_fit = compute_kappa(kk_fit, J, V, T)
        
        # --- Compute fluctuation F(t) and mean E(t) ---
        F, E, first_peak_x, first_zero_x = compute_F_E(ac, J, t)
        if plot_EF_flag:
            plot_F_E(t, F, first_peak_x, E, first_zero_x, n_plot=100000)
        
        # --- Compute FA and FD thermal conductivities ---
        tau_FA, kappa_FA, tau_FD, kappa_FD = compute_kappa_FA_FD(
            t, kap, first_peak_x, first_zero_x, timestep=dt
        )
        if plot_kappa_flag:
            plot_kappa_FA_FD(t, kap, kap_fit, tau_FA, kappa_FA, tau_FD, kappa_FD, n_plot=100000)
        
        # --- Compute thermal conductivity using Chen et al method ---
        kappa_C, kappa_F = compute_kappa_Chen(pars, J, V, T, tau_FA)
        
        # --- Save results in compressed .npz file ---
        np.savez_compressed(
            f"{dir_npz}/ws{window_size}_ss{step_size}_start{start}.npz",
            start=start, end=end,
            tau_FA=tau_FA, kappa_FA=kappa_FA,
            tau_FD=tau_FD, kappa_FD=kappa_FD,
            kap=kap, kapfit=kap_fit, pars=pars,
            kappa_C=kappa_C, kappa_F=kappa_F
        )
        
        # --- Free memory for next iteration ---
        del df, J, ac, ac_fit, kap, kap_fit
        gc.collect()


def plot_kappas_iso(base_dir="prod", directions=('x', 'y', 'z'),
                    timestep=0.001, stride=1, tplot=100):
    """
    Plot and average thermal conductivity curves obtained from Green-Kubo
    analysis along multiple spatial directions.

    This function loads thermal conductivity data stored in compressed
    .npz files for each direction, plots individual κ(t) curves, computes
    the running mean and standard deviation across all datasets, and
    visualizes FA characteristic points.

    A summary plot and a text file containing statistical information
    are saved to disk.

    Parameters
    ----------
    base_dir : str, optional
        Base directory containing directional subfolders with .npz files.
    directions : tuple of str, optional
        Spatial directions to include in the isotropic average.
    timestep : float, optional
        Simulation timestep.
    stride : int, optional
        Stride used in the original κ(t) computation.
    tplot : float, optional
        Maximum time shown in the plot.

    Returns
    -------
    None
    """

    # Lists used to collect FA / Chen / F results from all directions
    all_tau_FA = []
    all_kappa_FA = []
    all_kappa_C = []
    all_kappa_F = []

    # Effective timestep after striding
    timestep_eff = timestep * stride

    # Number of points shown in the time-dependent kappa plot
    n_points = int(tplot / timestep_eff)

    # Variables for online (Welford) mean and variance of kappa(t)
    kap_mean = np.zeros(n_points)
    kap_M2 = np.zeros(n_points)
    kap_count = 0

    # Time axis
    x = np.arange(n_points) * timestep_eff

    # Initialize figure
    plt.figure(figsize=(7, 4.5))

    # Loop over spatial directions
    for direction in directions:

        # Directory containing npz files for the current direction
        directory = os.path.join(base_dir, direction, "npz")
        npz_files = sorted(glob.glob(os.path.join(directory, "*.npz")))

        print(f"Reading {len(npz_files)} files from '{directory}'")

        # Loop over all windows saved for this direction
        for fname in npz_files:
            data = np.load(fname, allow_pickle=True)

            # Extract scalar quantities from file
            tau_FA = float(data["tau_FA"])
            kappa_FA = float(data["kappa_FA"])
            kappa_C = float(data["kappa_C"])
            kappa_F = float(data["kappa_F"])

            # Store values for statistical analysis
            all_tau_FA.append(tau_FA)
            all_kappa_FA.append(kappa_FA)
            all_kappa_C.append(kappa_C)
            all_kappa_F.append(kappa_F)

            print(f"kFA = {kappa_FA:.4f}")
            print(f"kF  = {kappa_F:.4f}")

            # Extract time-dependent thermal conductivity curve
            kap_raw = data["kap"]

            # Ensure consistent length with plotting window
            n_use = min(len(kap_raw), n_points)
            kap_vec = kap_raw[:n_use]

            # Plot individual kappa(t) curves with low opacity
            plt.plot(x[:n_use], kap_vec, alpha=0.3, lw=1)

            # Update running mean and variance (Welford algorithm)
            kap_count += 1
            delta = kap_vec - kap_mean[:n_use]
            kap_mean[:n_use] += delta / kap_count
            kap_M2[:n_use] += delta * (kap_vec - kap_mean[:n_use])

    # Compute and plot mean and standard deviation if more than one curve exists
    if kap_count > 1:
        kap_std = np.sqrt(kap_M2 / (kap_count - 1))
        plt.plot(x, kap_mean, lw=2, label="Mean κ(t)")
        plt.fill_between(
            x,
            kap_mean - kap_std,
            kap_mean + kap_std,
            alpha=0.2,
            label=r"±1$\sigma$"
        )

    # Convert collected lists into NumPy arrays
    taus_FA = np.array(all_tau_FA)
    kappas_FA = np.array(all_kappa_FA)
    kappas_C = np.array(all_kappa_C) if all_kappa_C else None
    kappas_F = np.array(all_kappa_F) if all_kappa_F else None

    # Scatter plot of FA characteristic points
    plt.scatter(
        taus_FA,
        kappas_FA,
        s=20,
        label="FA points",
        zorder=5
    )

    # Highlight mean FA point
    plt.scatter(
        np.mean(taus_FA),
        np.mean(kappas_FA),
        s=200,
        marker="*",
        label=(
            f"Mean FA "
            f"(τ={np.mean(taus_FA):.2f}, "
            f"κ={np.mean(kappas_FA):.2f})"
        ),
        zorder=10
    )

    # Axis labels and plot appearance
    plt.xlabel("Time")
    plt.ylabel("Thermal conductivity κ (W/m·K)")
    plt.xlim(0, tplot)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, which="both", ls="--", alpha=0.4)

    # Annotate combined directions
    plt.text(
        0.95, 0.05,
        f"Combined directions: {', '.join(directions)}",
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black"
        )
    )

    plt.tight_layout()

    # Save figure and summary statistics
    os.makedirs(base_dir, exist_ok=True)
    fig_path = os.path.join(base_dir, "kappa_summary_iso.png")
    txt_path = os.path.join(base_dir, "kappa_summary_iso.txt")

    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Write statistical summary to text file
    with open(txt_path, "w") as f:
        f.write("FA, tau_FA_mean, tau_FA_std, kappa_FA_mean, kappa_FA_std\n")
        f.write(
            f"{np.mean(taus_FA):.4f},"
            f"{np.std(taus_FA):.4f},"
            f"{np.mean(kappas_FA):.4f},"
            f"{np.std(kappas_FA):.4f}\n"
        )

        if kappas_C is not None:
            f.write(
                "Chen et al., "
                "kappa_C_mean, kappa_C_std, "
                "kappa_F_mean, kappa_F_std\n"
            )
            f.write(
                f"{np.mean(kappas_C):.4f},"
                f"{np.std(kappas_C):.4f},"
                f"{np.mean(kappas_F):.4f},"
                f"{np.std(kappas_F):.4f}\n"
            )

    # Explicit garbage collection for large datasets
    gc.collect()

    
# code by Antonio Sacco & docstring and comments by ChatGPT