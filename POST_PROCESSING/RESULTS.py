import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.io import read


def lighten_color(color, amount=0.5):
    """
    Lighten or darken a given color.

    Parameters
    ----------
    color : str or tuple
        Color specified as a hex string, color name, or RGB tuple.
    amount : float
        Lightening factor:
        - amount = 0   -> original color
        - amount = 1   -> white
        - amount < 0   -> darken towards black

    Returns
    -------
    tuple
        Lightened/darkened RGB color.
    """
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgb(c)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)


def results(base_dir, folders_list, x_values, x_label, 
            files_name="kappa_summary_iso.txt", out_name="results.png", 
            color_hex="#FF5733", lighten_amount=0.5, plot_FA=True, 
            plot_F=True, x_transform=None
):
    """
    Collect and plot thermal conductivity results from multiple simulation folders.

    This function reads FA and F thermal conductivity values from summary files,
    plots them as a function of a user-defined x-axis, and saves the resulting figure.

    Parameters
    ----------
    base_dir : str
        Base directory containing all simulation folders.
    folders_list : list of str
        List of subfolder names to read data from.
    x_values : list or array-like
        X-axis values corresponding to each folder (e.g. size, porosity, timestep).
    x_label : str
        Label for the x-axis.
    files_name : str, optional
        Name of the summary file inside each folder.
    out_name : str, optional
        Name of the output figure file.
    color_hex : str, optional
        Base color used for FA data.
    lighten_amount : float, optional
        Lightening factor used to generate the color for F data.
    plot_FA : bool, optional
        If True, plot κ_FA.
    plot_F : bool, optional
        If True, plot κ_F.
    x_transform : callable or None, optional
        Optional function applied to x_values (e.g. normalization).

    Returns
    -------
    None
    """

    #  Input validation 
    if len(folders_list) != len(x_values):
        raise ValueError("folders_list and x_values must have the same length.")

    # Apply optional transformation to x-axis values
    if x_transform is not None:
        x_vals = [x_transform(x) for x in x_values]
    else:
        x_vals = list(x_values)

    # Containers for extracted data
    k_FA, err_FA = [], []
    k_F, err_F = [], []

    #  Read data from files 
    for folder in folders_list:
        file_path = os.path.join(base_dir, folder, files_name)

        if not os.path.exists(file_path):
            print(f"⚠️ Missing file in {folder}, skipping.")
            continue

        with open(file_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

            # Expected format:
            # line 1 -> FA data
            # line 3 -> Chen/F data
            vals_FA = [float(v) for v in lines[1].split(",")]
            vals_F  = [float(v) for v in lines[3].split(",")]

            k_FA.append(vals_FA[2])
            err_FA.append(vals_FA[3])
            k_F.append(vals_F[2])
            err_F.append(vals_F[3])

    if not k_FA:
        print("No valid data found. Exiting.")
        return

    #  Plot 
    fig, ax = plt.subplots(figsize=(10, 5))

    color_FA = color_hex
    color_F  = lighten_color(color_hex, amount=lighten_amount)

    if plot_FA:
        ax.errorbar(
            x_vals, k_FA, yerr=err_FA,
            fmt='s', markersize=6, capsize=4,
            color=color_FA, label=r"$\kappa_{\mathrm{FA}}$"
        )

    if plot_F:
        ax.errorbar(
            x_vals, k_F, yerr=err_F,
            fmt='o', markersize=6, capsize=4,
            color=color_F, label=r"$\kappa_{\mathrm{F}}$"
        )

    #  Axis styling 
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Thermal conductivity $\kappa$ (W/mK)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, out_name), dpi=300)
    plt.close()


def compute_power_spectrum(signal, dt):
    """
    Compute the power spectrum of a time-dependent signal using FFT.

    The mean value of the signal is subtracted before computing the FFT
    to remove the zero-frequency component.

    Parameters
    ----------
    signal : array_like
        Input time series.
    dt : float
        Time step between consecutive data points.

    Returns
    -------
    freq : ndarray
        Positive frequency axis.
    spectrum : ndarray
        Power spectrum |FFT|^2 of the signal.
    """

    signal = np.asarray(signal)
    signal = signal - np.mean(signal)

    freq = np.fft.rfftfreq(len(signal), d=dt)
    spectrum = np.abs(np.fft.rfft(signal))**2

    return freq, spectrum


def compute_cn(atoms, cutoff_factor):
    """
    Compute the coordination number (CN) for each atom in an ASE Atoms object.

    The coordination number is calculated using ASE's NeighborList with
    natural cutoffs scaled by a user-defined factor.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure for which the coordination numbers are computed.
    cutoff_factor : float
        Multiplicative factor applied to the natural cutoffs.

    Returns
    -------
    cn : ndarray of int
        Array containing the coordination number of each atom.
    """
    n_atoms = len(atoms)

    # Ensure all atoms have the same chemical symbol
    atoms.set_chemical_symbols(["Au"] * n_atoms)

    # Build neighbor list using natural cutoffs
    cutoffs = natural_cutoffs(atoms, mult=cutoff_factor)
    nl = NeighborList(
        cutoffs=cutoffs,
        self_interaction=False,
        bothways=True
    )
    nl.update(atoms)

    # Compute coordination number for each atom
    cn = np.zeros(n_atoms, dtype=int)
    for i in range(n_atoms):
        indices, _ = nl.get_neighbors(i)
        cn[i] = len(indices)

    return cn


def exp_func(CN, a, b, c):
    """
    Exponential fit function for κ vs coordination number.

    Parameters
    ----------
    CN : float or array-like
        Coordination number (or deviation from bulk CN).
    a, b, c : float
        Fit parameters.

    Returns
    -------
    ndarray
        Function value a * exp(-b * CN) + c.
    """
    return a * np.exp(-b * CN) + c


def kappa_vs_cn(
    ax,
    base_dir_sim,
    base_dir_res,
    folders_list_sim,
    folders_list_res,
    label="",
    color_hex="#FF5733",
    lighten_amount=0.55,
    cutoff_factor=1.0,
    pillvalues=None,
    pillerrs=None
):
    """
    Plot thermal conductivity as a function of the average coordination number.

    For each configuration, the function:
    1. Reads the atomic structure (.xyz) and computes the coordination number.
    2. Reads the thermal conductivity results from `kappa_summary_iso.txt`.
    3. Saves CN and κ values to a text file.
    4. Plots κ_FA and κ_F as a function of CN.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object where the data will be plotted.
    base_dir_sim : str
        Base directory containing simulation folders with atomic structures.
    base_dir_res : str
        Base directory containing result folders with kappa summary files.
    folders_list_sim : list of str
        List of folders containing `.xyz` structure files.
    folders_list_res : list of str
        List of folders containing `kappa_summary_iso.txt`.
    label : str, optional
        Label used in the legend.
    color_hex : str, optional
        Base color for κ_FA data points.
    lighten_amount : float, optional
        Lightening factor for κ_F data points.
    cutoff_factor : float, optional
        Multiplicative factor for natural cutoffs in CN calculation.
    pillvalues : list or None, optional
        X-values used instead of CN (specific to pillar systems).
    pillerrs : list or None, optional
        Errors associated with pillvalues.

    Returns
    -------
    None
    """

    cn_mean_list = []
    k_FA_list, err_FA_list = [], []
    k_F_list,  err_F_list  = [], []


    # Loop over configurations
    for folder_sim, folder_res in zip(folders_list_sim, folders_list_res):

        #  Part 1: Coordination number 
        pattern = os.path.join(base_dir_sim, folder_sim, "*.xyz")
        xyz_files = sorted(glob.glob(pattern))

        if not xyz_files:
            print(f"⚠️ No .xyz file found in {pattern}")
            cn_mean_list.append(np.nan)
        else:
            try:
                atoms = read(xyz_files[0])
                cn = compute_cn(atoms, cutoff_factor=cutoff_factor)
                # ΔaCN = deviation from bulk fcc value (12)
                cn_mean_list.append(12.0 - np.mean(cn))
            except Exception:
                print(f"❌ Error reading structure in {xyz_files[0]}")
                cn_mean_list.append(np.nan)

        #  Part 2: Thermal conductivity 
        file_kappa = os.path.join(
            base_dir_res, folder_res, "kappa_summary_iso.txt"
        )

        if not os.path.exists(file_kappa):
            print(f"⚠️ Missing kappa file: {file_kappa}")
            k_FA_list.append(np.nan)
            err_FA_list.append(np.nan)
            k_F_list.append(np.nan)
            err_F_list.append(np.nan)
            continue

        with open(file_kappa, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        vals_FA = [float(x) for x in lines[1].split(",")]
        vals_F  = [float(x) for x in lines[3].split(",")]

        k_FA_list.append(vals_FA[2])
        err_FA_list.append(vals_FA[3])
        k_F_list.append(vals_F[2])
        err_F_list.append(vals_F[3])


    # Save CN–kappa data to file
    output_file = os.path.join(base_dir_res, "cn_kappa_output.txt")

    with open(output_file, "w") as f:
        f.write("CN_mean, k_FA, err_FA, k_F, err_F\n")
        for cn, kfa, erfa, kf, erf in zip(
            cn_mean_list, k_FA_list, err_FA_list, k_F_list, err_F_list
        ):
            f.write(f"{cn}, {kfa}, {erfa}, {kf}, {erf}\n")

    print(f"📄 File saved: {output_file}")


    # Plot
    color_FA = color_hex
    color_F  = lighten_color(color_hex, amount=lighten_amount)

    # Special case: pillar systems
    if (
        base_dir_sim == "../SIMS/PILLAR_SIMS"
        and pillvalues is not None
        and len(pillvalues) == len(k_FA_list)
    ):
        ax.errorbar(
            pillvalues, k_FA_list,
            xerr=pillerrs, yerr=err_FA_list,
            fmt='s', capsize=4, markersize=6,
            color=color_FA, label=label
        )
    else:
        ax.errorbar(
            cn_mean_list, k_FA_list,
            yerr=err_FA_list,
            fmt='s', capsize=4, markersize=6,
            color=color_FA, label=label
        )

    ax.errorbar(
        cn_mean_list, k_F_list,
        yerr=err_F_list,
        fmt='o', capsize=4, markersize=6,
        color=color_F
    )

    ax.set_xlabel(r"$\Delta a\,\mathrm{CN}$")
    ax.set_ylabel(r"$\kappa$ (W/mK)")
    ax.grid(True, ls='--', alpha=0.4)


def cn_vs_porosity(ax, base_dir, folders_list, x_values, label,
                   color_hex="#FF5733", lighten_amount=0.55,
                   cutoff_factor=1, plot_cn=True):
    """
    Compute and plot the mean coordination number (CN) of atoms versus porosity.

    This function automatically searches for a single *.xyz file in each folder, 
    reads the atomic structure using ASE, computes the coordination number for 
    each atom, and calculates the mean CN. Optionally, it plots mean CN against 
    the normalized porosity values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to plot the data.
    base_dir : str
        Base directory containing the simulation folders.
    folders_list : list of str
        List of folder names where each folder contains one *.xyz file.
    x_values : list of float
        List of porosity values corresponding to each folder.
    label : str
        Label for the plot legend.
    color_hex : str, optional
        Color of the plotted line and markers, by default "#FF5733".
    lighten_amount : float, optional
        Amount to lighten the color (not used directly here but kept for consistency), by default 0.55.
    cutoff_factor : float, optional
        Multiplicative factor for computing the coordination number cutoff, by default 1.
    plot_cn : bool, optional
        Whether to plot the mean CN or just return the values, by default True.

    Returns
    -------
    None
        The function plots the CN vs porosity on the provided axes and modifies the axes in-place.
    """

    # Check that folders_list and x_values have the same length
    if len(folders_list) != len(x_values):
        raise ValueError("folders_list and x_values must have the same length.")

    cn_mean_list = []  # store mean coordination numbers for each folder

    # --- Loop through each folder to compute mean CN ---
    for folder in folders_list:

        # Search for any *.xyz files in the folder
        pattern = os.path.join(base_dir, folder, "*.xyz")
        data_files = sorted(glob.glob(pattern))

        if len(data_files) == 0:
            print(f"⚠️ No .xyz file found in {folder}")
            cn_mean_list.append(np.nan)
            continue

        # Take the first found file
        file_path = data_files[0]

        # Read atomic structure using ASE
        try:
            atoms = read(file_path)
        except:
            print(f"Error reading {file_path}")
            cn_mean_list.append(np.nan)
            continue

        # Compute coordination number for all atoms
        cn = compute_cn(atoms, cutoff_factor=cutoff_factor)
        cn_mean = np.mean(cn)  # compute mean CN
        cn_mean_list.append(cn_mean)

    # --- Normalize x_values by bulk atom number ---
    bulk_atoms = 5324
    x_values_normalized = [x / bulk_atoms for x in x_values]

    # --- Plot mean CN vs normalized porosity ---
    if plot_cn:
        ax.plot(
            x_values_normalized,
            cn_mean_list,
            "o-",  # line with circle markers
            color=color_hex,
            markersize=6,
            label=label
        )

    # Set axis labels and grid
    ax.set_xlabel("Porosity")
    ax.set_ylabel("Mean coordination number")
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()


