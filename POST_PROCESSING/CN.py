import os                                               
from tqdm import tqdm # type: ignore             
from ase import Atoms   
from ase.io import read              
from ase.neighborlist import NeighborList, natural_cutoffs
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_cn(atoms, cutoff_factor=1.0):
    """
    Calculates the coordination number for each atom in an ASE Atoms object.
    
    atoms: ASE Atoms object
    cutoff_factor: multiplicative factor for the natural cutoffs (default 1.0)
    """

    n_atoms = len(atoms)                             # Total number of atoms in the Atoms object
    atoms.set_chemical_symbols(["Au"] * n_atoms)     # Set all atomic symbols to "Au" (needed for natural_cutoffs)
    
    # Compute natural cutoffs for each atom and scale by cutoff_factor
    cutoffs = natural_cutoffs(atoms, mult=cutoff_factor)
    # natural_cutoffs returns an array of cutoff radii for each atom
    # 'cutoff_factor' scales the natural values (e.g., 1.2 = 20% larger)

    # Create a NeighborList:
    # - cutoffs: search radius for each atom
    # - self_interaction=False: do not count the atom itself
    # - bothways=True: neighbors are recorded symmetrically (i is neighbor of j and j of i)
    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)

    nl.update(atoms)                                 # Build the neighbor list for the current atomic configuration

    cn = np.zeros(n_atoms, dtype=int)                # Array to store coordination numbers for each atom

    for i in range(n_atoms):                         # Loop over all atoms
        indices, offsets = nl.get_neighbors(i)      # Get neighbor indices and cell offsets (periodicity)
        cn[i] = len(indices)                        # Coordination number = number of neighbors

    return cn                                       # Return array of coordination numbers
                                  

def plot_cn_xyz_binned(atoms, cn, bins, filename):
    """
    Plot the average coordination number (CN) versus x, y, z coordinates using bins.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers for each atom
    bins: number of bins or array of bin edges
    filename: string used for the plot title
    """

    pos = atoms.get_positions()  # Get atomic positions as a numpy array

    # Create a figure with 3 subplots for x, y, z coordinates
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("cn profiles of " + filename)  # Overall title
    labels = ['x', 'y', 'z']

    for i, ax in enumerate(axes):
        coord = pos[:, i]  # Select coordinate array (x, y, or z)
        
        # Weighted histogram: sum of CNs in each bin
        bin_means, bin_edges = np.histogram(coord, bins=bins, weights=cn)
        # Count of atoms in each bin
        bin_counts, _ = np.histogram(coord, bins=bin_edges)
        # Compute average CN per bin, avoid division by zero
        bin_avg = bin_means / np.maximum(bin_counts, 1)
        # Compute bin centers for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(bin_centers, bin_avg, color='red', lw=2)  # Plot mean CN vs coordinate
        ax.set_xlabel(f"{labels[i]} [Å]")               # Label x-axis
        ax.set_ylabel("Mean CN")                        # Label y-axis
        ax.grid(True)                                   # Add grid for readability

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()          # Display the plot


def mean_cn_in_z_range(atoms, cn, z_min, z_max):
    """
    Calculate the mean coordination number (CN) for atoms with z-coordinate 
    between z_min and z_max.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers for each atom
    z_min, z_max: lower and upper bounds for z-coordinate (Å)
    """

    z = atoms.get_positions()[:, 2]  # Extract z-coordinates of all atoms

    # Create a boolean mask selecting atoms within the z range
    mask = (z >= z_min) & (z <= z_max)
    cn_selected = cn[mask]  # Apply mask to CN array to select relevant atoms

    mean_cn = cn_selected.mean()  # Compute mean CN in the selected z range
    print(f"Mean CN between z = {z_min:.2f} Å and z = {z_max:.2f} Å: {mean_cn:.3f}")

    return mean_cn  # Return the mean CN


def save_file_with_cn(atoms, cn, out_file):
    """
    Save an .xyz file with an extra column containing the coordination number (CN) for each atom.
    
    atoms: ASE Atoms object
    cn: array of coordination numbers
    out_file: output filename (string)
    """

    n_atoms = len(atoms)                     # Total number of atoms
    pos = atoms.get_positions()              # Atomic positions (Nx3 array)
    symbols = atoms.get_chemical_symbols()  # List of chemical symbols for each atom

    # Open the output file for writing
    with open(out_file, "w") as f:
        f.write(f"{n_atoms}\n")                     # First line: number of atoms
        f.write("XYZ file with Coordination Number\n")  # Second line: comment/header

        # Loop over atoms and write symbol, coordinates, and CN
        for s, (x, y, z), c in zip(symbols, pos, cn):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f} {c}\n")
            # Format coordinates with 6 decimal places, append CN

    print(f"File saved to: {out_file}")  # Print confirmation


def analyze_cn_from_file(filename, cutoff_factor, bins, out_file):
    """
    Read a data file, compute coordination numbers (CN), print statistics,
    plot CN profiles, and optionally save an XYZ file with CN.

    filename: input data file
    cutoff_factor: multiplicative factor for CN calculation
    bins: number of bins or array of bin edges for plotting
    out_file: if True, save an XYZ file with CN appended
    """

    # Read atomic configuration from LAMMPS data file
    atoms = read(filename, format="lammps-data", atom_style='atomic')

    # Compute coordination numbers
    cn = compute_cn(atoms, cutoff_factor=cutoff_factor)

    # Print basic statistics
    print(f"Number of atoms: {len(atoms)}")
    print(f"Mean CN: {cn.mean():.3f}")
    print(f"CN min/max: {cn.min()} / {cn.max()}")
    # Compute mean CN for atoms between two slabs (z = 8.1 Å to 71.5 Å)
    print(f"Mean CN between slabs: {mean_cn_in_z_range(atoms, cn, 8.1, 71.5)}")

    # Plot CN profiles along x, y, z
    plot_cn_xyz_binned(atoms, cn, bins, filename)

    # Optionally save an XYZ file with CN column
    if out_file == True:
        outname = filename.replace("data", "xyzcn")  # Replace extension for output
        save_file_with_cn(atoms, cn, outname)


def process_large_dump(path, infile, out_dir, outfile, cutoff_factor=1, skip=1):
    """
    Processes a large LAMMPS dump file frame by frame, computes coordination numbers (CN),
    and writes a new dump file with CN appended to each atom.
    
    path: directory containing the input dump
    infile: input dump filename
    out_dir: directory to save output
    outfile: output filename
    cutoff_factor: multiplicative factor for CN calculation
    skip: process every 'skip' frame (default 1 = all frames)
    """

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)                        # Make the output directory
        print(f"Created output directory: {out_dir}")

    infile_path = os.path.join(path, infile)        # Full path to input file
    outfile_path = os.path.join(out_dir, outfile)   # Full path to output file

    print(f"Reading from: {infile_path} & Writing to: {outfile_path}\n")

    # --- Count number of frames ---
    n_frames = 0
    with open(infile_path, "r") as fcount:
        for line in fcount:
            if "ITEM: TIMESTEP" in line:            # Each LAMMPS frame starts with this line
                n_frames += 1                       # Increment frame count

    # --- Process file with tqdm progress bar ---
    with open(infile_path, "r") as fin, open(outfile_path, "w") as fout:
        frame = 0
        # tqdm shows progress bar with known total = n_frames
        with tqdm(total=n_frames, desc="Processing LAMMPS dump", unit="frame") as pbar:
            while True:
                line = fin.readline()               # Read a line from input file
                if not line:
                    break                           # End of file -> exit loop

                if "ITEM: TIMESTEP" in line:        # Start of a new LAMMPS frame
                    timestep = int(fin.readline().strip())  # Read the timestep number
                    fout.write("ITEM: TIMESTEP\n")
                    fout.write(f"{timestep}\n")            # Write header and timestep to output

                    # --- Number of atoms ---
                    fin.readline()                         # Skip "ITEM: NUMBER OF ATOMS" line
                    n_atoms = int(fin.readline().strip())  # Read number of atoms
                    fout.write("ITEM: NUMBER OF ATOMS\n")
                    fout.write(f"{n_atoms}\n")             # Write number of atoms to output

                    # --- Box bounds ---
                    box_header = fin.readline()            # Read box bounds header
                    fout.write(box_header)                 # Write header to output
                    box = []
                    for _ in range(3):                    # Read three lines defining x, y, z bounds
                        bounds = list(map(float, fin.readline().split()))
                        box.append(bounds)                # Store bounds
                        fout.write(f"{bounds[0]} {bounds[1]}\n")  # Write bounds to output

                    # --- Atom section header ---
                    atoms_header = fin.readline().strip()  # Read atom data header
                    fout.write(atoms_header + " cn\n")     # Write header adding a "cn" column

                    # --- Atom data ---
                    data_lines = [fin.readline() for _ in range(n_atoms)]
                    # Read n_atoms lines containing atomic data

                    data = np.array([list(map(float, l.split())) for l in data_lines])
                    # Convert lines to a numeric array
                    pos = data[:, 1:4]                    # Assume columns 1,2,3 are x, y, z
                    box_lengths = [b[1] - b[0] for b in box]
                    # Compute cell lengths from bounds
                    atoms = Atoms("Au" * n_atoms, positions=pos, cell=box_lengths, pbc=True)
                    # Create ASE Atoms object:
                    # - symbols: "Au" repeated n_atoms
                    # - positions: extracted positions
                    # - cell: x, y, z lengths
                    # - pbc=True: enable periodic boundary conditions

                    if frame % skip == 0:
                        cn = compute_cn(atoms, cutoff_factor)  # Compute CN every 'skip' frames
                    else:
                        cn = np.zeros(n_atoms)                 # If skipped, CN = 0

                    for i in range(n_atoms):
                        vals = " ".join(f"{v:g}" for v in data[i])
                        # Recompose original atom line in compact format
                        fout.write(f"{vals} {int(cn[i])}\n")
                        # Write line with CN appended

                    frame += 1
                    pbar.update(1)  # Update progress bar after processing the frame


def get_acnt_dump(path, infile, cutoff_factor=1, skip=1):
    """
    Compute the average coordination number (aCN) for each frame of a LAMMPS dump file.

    The function reads a LAMMPS dump file frame by frame, constructs an ASE `Atoms`
    object for selected frames, computes the coordination number for each atom,
    and stores the mean coordination number per frame.

    Parameters
    ----------
    path : str
        Directory containing the dump file.
    infile : str
        Name of the LAMMPS dump file.
    cutoff_factor : float, optional
        Multiplicative factor applied to the natural cutoff when computing
        coordination numbers, by default 1.
    skip : int, optional
        Process only one frame every `skip` frames (e.g. skip=10 processes
        frames 0, 10, 20, ...), by default 1.

    Returns
    -------
    np.ndarray
        Array containing the mean coordination number for each processed frame.
    """

    # Full path to the input dump file
    infile_path = os.path.join(path, infile)

    # List to store mean coordination number per frame
    cn_mean_list = []

    # --------------------------------------------------
    # Count total number of frames in the dump file
    # --------------------------------------------------
    n_frames = 0
    with open(infile_path, "r") as fcount:
        for line in fcount:
            if "ITEM: TIMESTEP" in line:
                n_frames += 1

    # --------------------------------------------------
    # Read and process the dump file frame by frame
    # --------------------------------------------------
    with open(infile_path, "r") as fin:
        frame = 0  # frame counter

        # Progress bar over total number of frames
        with tqdm(total=n_frames, desc="obtaining aCN per frame", unit="frame") as pbar:
            while True:
                line = fin.readline()
                if not line:
                    break  # end of file

                if "ITEM: TIMESTEP" in line:

                    # Read timestep value
                    timestep = int(fin.readline().strip())

                    # ITEM: NUMBER OF ATOMS
                    fin.readline()
                    n_atoms = int(fin.readline().strip())

                    # ITEM: BOX BOUNDS
                    fin.readline()
                    box = []
                    for _ in range(3):
                        bounds = list(map(float, fin.readline().split()))
                        box.append(bounds)

                    # ITEM: ATOMS ...
                    fin.readline()

                    # ------------------------------------------
                    # Skip frame if not selected
                    # ------------------------------------------
                    if frame % skip != 0:
                        # Skip atom lines without reading data
                        for _ in range(n_atoms):
                            fin.readline()
                        frame += 1
                        pbar.update(1)
                        continue

                    # ------------------------------------------
                    # Read atomic data for selected frame
                    # ------------------------------------------
                    data = np.array([
                        list(map(float, fin.readline().split()))
                        for _ in range(n_atoms)
                    ])

                    # Extract atomic positions (assumes x,y,z are columns 1:4)
                    pos = data[:, 1:4]

                    # Compute box lengths from bounds
                    box_lengths = [b[1] - b[0] for b in box]

                    # Build ASE Atoms object
                    atoms = Atoms(
                        "Au" * n_atoms,
                        positions=pos,
                        cell=box_lengths,
                        pbc=True
                    )

                    # Compute coordination number for each atom
                    cn = compute_cn(atoms, cutoff_factor)

                    # Compute mean coordination number for this frame
                    cn_mean = np.mean(cn)
                    cn_mean_list.append(cn_mean)

                    frame += 1
                    pbar.update(1)

    # Return mean CN values as a NumPy array
    return np.array(cn_mean_list)


def compare_profiles_in_z(filenames, cutoff_factor=1.2, bins=30):
    """
    Compare coordination-number and atomic-density profiles along the z direction
    for multiple atomic configurations.

    For each input structure, the function computes:
    - the average coordination number in z-bins
    - the number of atoms in the same z-bins

    Both quantities are plotted as a function of z to allow direct comparison
    between different configurations (e.g. different simulation times).

    Parameters
    ----------
    filenames : list of str
        List of input structure files readable by ASE (e.g. .xyz).
        All structures must be comparable along the z direction.
    cutoff_factor : float, optional
        Multiplicative factor applied to ASE natural cutoffs when computing
        the coordination number.
    bins : int, optional
        Number of bins used to discretize the z axis.

    Returns
    -------
    None
        The function produces and displays a matplotlib figure.
    """

    # Create a figure with two vertically stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(5, 5))

    n_files = len(filenames)

    # Horizontal offsets used to separate bars belonging to different files
    offsets = np.linspace(-0.4, 0.4, n_files)

    # Define common z-bins using the first configuration
    atoms0 = read(filenames[0])
    z0 = atoms0.get_positions()[:, 2]

    # Compute histogram edges once to ensure identical binning
    _, edges = np.histogram(z0, bins=bins)

    # Bin centers and width
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    bar_width = bin_width * 0.35

    # Labels associated with each configuration (user-defined assumption)
    labels = ["0ns", "100ns"]

    # Loop over input structures
    for idx_f, fname in enumerate(filenames):

        # Read atomic configuration
        atoms = read(fname)
        pos = atoms.get_positions()[:, 2]

        # Compute coordination number for each atom
        cn = compute_cn(atoms, cutoff_factor)

        # Histogram of summed coordination numbers per z-bin
        cn_sum, _ = np.histogram(pos, bins=edges, weights=cn)

        # Histogram of atom counts per z-bin
        counts, _ = np.histogram(pos, bins=edges)

        # Average coordination number per bin
        # (protected against division by zero)
        cn_avg = cn_sum / np.maximum(counts, 1)

        # Plot average coordination number profile
        axes[0].plot(
            centers,
            cn_avg,
            marker="o",
            markeredgecolor="black",
            lw=1,
            label=labels[idx_f]
        )

        # Plot atomic density (counts) profile
        axes[1].bar(
            centers + offsets[idx_f] * bar_width,
            counts,
            width=bar_width,
            edgecolor="black",
            label=labels[idx_f],
            alpha=1
        )

    # Axis labels, legends, and layout
    axes[0].set_xlabel(r"z [$\AA$]")
    axes[0].set_ylabel("CNL")
    axes[0].legend()

    axes[1].set_xlabel(r"z [$\AA$]")
    axes[1].set_ylabel("Atoms count")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# by Antonio Sacco & commented by ChatGPT