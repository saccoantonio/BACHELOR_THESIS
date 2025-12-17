# Bachelor's Thesis Repository

This repository contains the material developed for my Bachelor’s thesis at Università degli Studi di Milano (UniMi), including simulation input files, data post-processing scripts, the LaTeX sources of the written thesis and summary, and finally the presentation.

---

## Thesis Overview

In this thesis, I studied thermal transport in different gold systems, focusing on how porosity affect thermal conductivity, from the perfect bulk to nanopillars and deposited nanofilms.

The analysis was carried out using equilibrium molecular dynamics simulations with LAMMPS and the Green–Kubo formalism. Atomic structures were generated with ASE, while post-processing was performed with custom Python scripts to compute thermal conductivity and structural descriptors such as porosity and average coordination number.

The results show a monotonic decrease in thermal conductivity with increasing porosity. However, porosity alone is not sufficient to predict thermal conductivity, as systems with the same porosity can exhibit significantly different values. The average coordination number provides a more accurate and universal descriptor of thermal conductivity and is applicable even to systems for which porosity is not well defined. As the coordination number decreases, thermal conductivity consistently decreases across all the investigated systems, allowing us to conclude that the coordination number dominates the phononic contribution to thermal transport in gold systems.

---

## Repository Structure

The repository is organized into the following main directories:

### POST_PROCESSING
This folder contains Python scripts and notebooks used for:
- Computation of thermal conductivity via the Green-Kubo formalism &rarr; GK.py
- Generation and manipulation of atomic configurations &rarr; CREATOR.py
- Calculation of geometrical descriptors &rarr; CN.py
- Final production of results used in the thesis &rarr; RESULTS.py

---

### LAMMPS
This folder contains the main **LAMMPS input file** used to run the Molecular Dynamics simulations. The simulations consist of three phases:
- Thermalisation at a target temperature of 300 K.
- Equilibration, to eliminate transient effects related to the application of the thermostat.
- Production, where we record all the quantities necessary for calculating thermal conductivity.

---

### THESIS
LaTeX source files of the full written thesis, including:
- Main document
- Figures
- Bibliography
- Output PDF

---

### SUMMARY
LaTeX source files for the thesis summary and output PDF.

---

### PRESENTATION
PDF files of the slides used for the thesis discussion.

---

## Notes

- The repository is structured to ensure **reproducibility** of the simulations and analysis.
- Some paths inside input files may need to be adapted depending on the local system configuration.
- Large raw simulation outputs are not included.

---

## Author

Antonio Sacco  
Bachelor’s Thesis Project  
