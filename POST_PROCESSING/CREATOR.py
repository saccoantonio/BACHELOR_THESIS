from ase.lattice.cubic import FaceCenteredCubic
import numpy as np
from ase.io import write, read
from scipy.spatial import cKDTree


def create_bulk(n):
    """Creates a bulk FCC gold (Au) crystal of size n x n x n unit cells with periodic boundary conditions."""

    # Parameters
    symbol = "Au"  # Atomic symbol for gold
    a0 = 4.078     # Lattice constant of Au FCC in Ångstroms

    # Create a bulk FCC structure of gold
    bulk = FaceCenteredCubic(
        directions=[[1, 0, 0],   # Define orientation of the lattice vectors
                    [0, 1, 0],
                    [0, 0, 1]],
        size=(n, n, n),           # Number of unit cells in x, y, z directions
        symbol=symbol,            # Element type
        pbc=True                   # Enable periodic boundary conditions
    )

    final_config = bulk         # Copy the bulk structure to final configuration
    final_config.set_cell(bulk.get_cell())  # Ensure the cell vectors are explicitly set
    final_config.set_pbc([True, True, True])  # Make sure periodic boundary conditions are on in all directions

    #print(f"Numero di atomi: {len(final_config)}")  # Optional: print the number of atoms
    return final_config        # Return the final bulk configuration


def create_vacancies(n_remove):
    """
    Create random point vacancies in a bulk FCC gold system.

    A given number of atoms is randomly removed from the bulk,
    without any geometric constraint.

    Parameters
    ----------
    n_remove : int
        Number of atoms to remove randomly.

    Returns
    -------
    None
        A LAMMPS data file named 'vacancies<n_remove>.data' is written.
    """

    # Number of unit cells
    n = 11

    # Generate bulk structure
    bulk = create_bulk(n)

    # Indices of all atoms (entire bulk)
    central_indices = [i for i in range(len(bulk))]

    # Randomly select atoms to remove (without replacement)
    remove_from_central = np.random.choice(
        central_indices, n_remove, replace=False
    )

    # Keep only atoms that are not removed
    final_indices = list(
        set(range(len(bulk))) - set(remove_from_central)
    )

    # Build final system
    final_system = bulk[final_indices]

    # Write LAMMPS data file
    write(
        f'vacancies{n_remove}.data',
        final_system,
        format='lammps-data',
        atom_style='atomic'
    )


def create_holes(n_remove, radius, seed):
    """
    Create clustered holes by removing atoms within spherical regions
    under periodic boundary conditions.

    The algorithm:
    - considers all atoms as candidates
    - builds periodic replicas to handle PBC correctly
    - randomly selects centers
    - removes atoms within a given radius until a target number is reached

    Parameters
    ----------
    n_remove : int
        Target number of atoms to remove.
    radius : float
        Radius of each spherical hole (Å).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    None
        A LAMMPS data file named 'holes_<n_remove>_<radius>.data' is written.
    """

    # Number of unit cells
    n = 11

    # FCC Au lattice constant (Å)
    a0 = 4.078

    # Generate bulk structure
    bulk = create_bulk(n)
    pos = bulk.get_positions()

    # Simulation cell as a 3x3 array
    cell = np.asarray(bulk.get_cell())

    # Set random seed
    np.random.seed(seed)

    # ---- central region: here the whole system ----
    central_indices = [i for i in range(len(bulk))]
    if len(central_indices) == 0:
        raise RuntimeError(
            "No central atoms found: check z-limits or geometry."
        )

    central_pos = pos[central_indices]
    n_central = len(central_indices)

    # Target number of atoms to remove
    n_target = n_remove
    if n_target <= 0:
        print("n_target = 0. Nothing will be removed.")
        return bulk

    # ---- build periodic shifts (only for active PBC directions) ----
    pbc = (True, True, True)
    ranges = [
        (-1, 0, 1) if pbc[i] else (0,)
        for i in range(3)
    ]

    # All combinations of periodic image shifts
    shifts = np.array(
        [[i, j, k]
         for i in ranges[0]
         for j in ranges[1]
         for k in ranges[2]]
    )

    # Convert integer shifts into real-space displacements
    disp = shifts @ cell

    # ---- replicate central positions for PBC handling ----
    # Shape: (n_shifts * n_central, 3)
    replicas = (
        central_pos[None, :, :] + disp[:, None, :]
    ).reshape(-1, 3)

    # Mapping from replica index to original atom index
    mapping = np.tile(central_indices, len(shifts))

    # ---- KDTree for fast neighbor search under PBC ----
    tree = cKDTree(replicas)

    # ---- iterate over random candidate centers ----
    order = np.random.permutation(n_central)
    to_remove = set()

    for local_center_idx in order:
        if len(to_remove) >= n_target - 11:
            break

        # Candidate center position
        center = central_pos[local_center_idx]

        # Find all replica atoms within the given radius
        neighbors = tree.query_ball_point(center, radius)
        if not neighbors:
            continue

        # Convert replica indices to global atom indices
        bad_globals = {mapping[i] for i in neighbors}
        to_remove.update(bad_globals)

    # ---- build final system ----
    final_indices = [
        i for i in range(len(bulk))
        if i not in to_remove
    ]
    final_system = bulk[final_indices]

    # Write output file
    outname = f'holes_{n_remove}_{radius:.1f}.data'
    write(outname, final_system, format='lammps-data', atom_style='atomic')

    print(
        f"Saved {outname} — removed {len(to_remove)} atoms "
        f"({100 * len(to_remove) / n_central:.2f}% of the central region)"
    )


def create_void(radius):
    """
    Create a spherical void by removing atoms inside a given radius
    from the center of the simulation box.

    Parameters
    ----------
    radius : float
        Radius of the spherical void (Å).

    Returns
    -------
    None
        A LAMMPS data file named 'void_radius<radius>.data' is written.
    """

    # Number of unit cells
    n = 11

    # FCC Au lattice constant (Å)
    a0 = 4.078

    # Generate bulk structure
    bulk = create_bulk(n)
    pos = bulk.get_positions()

    # Indices of all atoms
    indices = [i for i in range(len(bulk))]

    # ---- define sphere center ----
    box = bulk.get_cell().array
    center_x = box[0, 0] / 2
    center_y = box[1, 1] / 2
    center_z = box[2, 2] / 2
    center = np.array([center_x, center_y, center_z])

    # ---- select atoms inside the sphere ----
    remove = []
    for i in indices:
        r_vec = pos[i] - center
        dist = np.linalg.norm(r_vec)
        if dist < radius:
            remove.append(i)

    # Build final system without the void atoms
    final_indices = list(
        set(range(len(bulk))) - set(remove)
    )
    final_system = bulk[final_indices]

    # Write output file
    write(
        f'void_radius{radius}.data',
        final_system,
        format='lammps-data',
        atom_style='atomic'
    )

    print(
        f"With radius = {radius}, {len(remove)} atoms have been removed "
        f"({len(remove) / len(indices) * 100:.2f}%)."
    )


def create_pillar(side1, side2):
    """
    Create a rectangular pillar carved from a bulk FCC gold crystal.

    The function selects atoms inside a rectangular cross-section
    centered in the xy-plane and keeps periodic boundary conditions
    in all directions.

    Parameters
    ----------
    side1 : float
        Width of the pillar along the x direction (Å).
    side2 : float
        Width of the pillar along the y direction (Å).

    Returns
    -------
    None
        A LAMMPS data file named 'bulkpillar.data' is written to disk.
    """

    # Number of unit cells used to build the bulk system
    n = 15

    # FCC Au lattice constant (Å)
    a0 = 4.078

    # Generate bulk structure
    bulk = create_bulk(n)
    positions = bulk.get_positions()

    # Compute center of the bulk system
    bulk_center = positions.mean(axis=0)

    # Half-widths of the pillar in x and y
    dx = side1
    dy = side2

    # Select atoms inside the rectangular pillar cross-section
    column_indices = [
        i for i, atom in enumerate(bulk)
        if -dx / 2 < atom.position[0] - bulk_center[0] < dx / 2 and
           -dy / 2 + 2 < atom.position[1] - bulk_center[1] < dy / 2
    ]

    # Extract the pillar configuration
    final_config = bulk[column_indices]

    # Assign original cell and periodic boundary conditions
    final_config.set_cell(bulk.get_cell())
    final_config.set_pbc([True, True, True])

    print(f"Number of atoms in the final configuration: {len(column_indices)}")

    # Write LAMMPS data file
    write("bulkpillar.data", final_config, format="lammps-data", atom_style="atomic")


def create_column(radius):
    """
    Create a cylindrical column carved from a bulk FCC gold crystal.

    The column is defined by selecting atoms within a given radius
    from the center in the xy-plane.

    Parameters
    ----------
    radius : float
        Radius of the cylindrical column (Å).

    Returns
    -------
    None
        A LAMMPS data file named 'column.data' is written to disk.
    """

    # Number of unit cells for the bulk system
    n = 15

    # FCC Au lattice constant (Å)
    a0 = 4.078

    # Generate bulk structure
    bulk = create_bulk(n)
    positions = bulk.get_positions()

    # Compute center of the bulk system
    bulk_center = positions.mean(axis=0)

    # Select atoms inside the cylindrical region in the xy-plane
    column_indices = [
        i for i, atom in enumerate(bulk)
        if np.sqrt(
            (atom.x - bulk_center[0])**2 +
            (atom.y - bulk_center[1])**2
        ) < radius
    ]

    # Extract the column configuration
    final_config = bulk[column_indices]

    # Assign original cell and periodic boundary conditions
    final_config.set_cell(bulk.get_cell())
    final_config.set_pbc([True, True, True])

    print(f"Number of atoms in the final configuration: {len(column_indices)}")

    # Write LAMMPS data file
    write("column.data", final_config, format="lammps-data", atom_style="atomic")


def nano_pillar(filename):
    """
    Insert a nanopillar structure into a bulk FCC gold system.

    The function:
    - creates a bulk slab
    - removes a central region along z
    - inserts a nanopillar read from file
    - shifts the upper slab to avoid overlaps
    - updates the simulation cell accordingly

    Parameters
    ----------
    filename : str
        XYZ file containing the nanopillar structure to be inserted.

    Returns
    -------
    None
        A LAMMPS data file named 'my<filename>.data' is written to disk.
    """

    # Number of unit cells for the bulk system
    n = 15

    # FCC Au lattice constant (Å)
    a0 = 4.078

    # Generate bulk structure
    bulk = create_bulk(n)
    positions = bulk.get_positions()

    # Cross-section size for the pillar region (Å)
    dx = dy = 16

    # Compute bulk center
    bulk_center = positions.mean(axis=0)

    # Define z-limits for removing the central region
    zmin = bulk_center[2] - 27.5
    zmax = bulk_center[2] + 27.5

    # Select upper slab atoms within the pillar cross-section
    upper_indices = [
        i for i, atom in enumerate(bulk)
        if atom.position[2] > zmax and
           -dx / 2 < atom.position[0] - bulk_center[0] < dx / 2 and
           -dy / 2 < atom.position[1] - bulk_center[1] < dy / 2
    ]

    # Select lower slab atoms within the pillar cross-section
    lower_indices = [
        i for i, atom in enumerate(bulk)
        if atom.position[2] < zmin and
           -dx / 2 < atom.position[0] - bulk_center[0] < dx / 2 and
           -dy / 2 < atom.position[1] - bulk_center[1] < dy / 2
    ]

    # Read nanopillar structure
    nanopillar = read(filename)

    # Compute centers
    pillar_center = nanopillar.get_positions().mean(axis=0)
    target_center = np.array([
        bulk_center[0],
        bulk_center[1],
        0.5 * (zmin + zmax)
    ])

    # Translate nanopillar into the central gap
    shift = target_center - pillar_center
    nanopillar.translate(shift)

    # Shift upper slab upward to avoid overlap with nanopillar
    max_nano_z = nanopillar.get_positions()[:, 2].max()
    low_upslab_z = bulk.positions[upper_indices, 2].min()
    dz = low_upslab_z - max_nano_z - 2.04

    bulk.positions[upper_indices, 2] -= dz

    # Combine lower slab, nanopillar, and upper slab
    final_config = bulk[upper_indices] + nanopillar + bulk[lower_indices]

    # Update simulation cell along z
    cell = bulk.get_cell()
    cell[2, 2] -= dz

    final_config.set_cell(cell)
    final_config.set_pbc([True, True, True])

    name = filename.split(".")[0]
    print(f"Total number of atoms in final configuration: {len(final_config)}")

    # Write final LAMMPS data file
    write(f"my{name}.data", final_config, format="lammps-data", atom_style="atomic")


def lammps_data_to_xyz(input_file):
    """
    Convert a LAMMPS data file into an XYZ file using ASE.

    The function reads a LAMMPS data file (atom_style = atomic),
    assigns all atoms the chemical symbol 'Au', and writes the
    resulting structure to an XYZ file with the same base name.

    Parameters
    ----------
    input_file : str
        Path to the input LAMMPS data file.

    Returns
    -------
    None
        The function writes an XYZ file to disk and does not return anything.
    """

    # Read the LAMMPS data file using ASE
    atoms = read(input_file, format="lammps-data", atom_style="atomic")

    # Assign the chemical symbol 'Au' to all atoms
    # (useful when the data file does not specify element types explicitly)
    atoms.set_chemical_symbols(["Au"] * len(atoms))

    # Generate output filename by replacing 'data' with 'xyz'
    output_file = input_file.replace("data", "xyz")

    # Write the atomic structure to an XYZ file
    write(output_file, atoms, format="xyz")


# by Antonio Sacco & commented by ChatGPT