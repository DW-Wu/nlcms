# Diffuse-interface LdG model

This is a **N**ematic **L**iquid **C**rystal model (with **M**odified bulk energy) implemented with **S**pectral method. 

## Program structure

`nlc_state.py` and `nlc_func.py` are infrastructural programs.
`nlc_plot.py` visualizes the NLC state with methods described in [Hu et al. 2016](https://www.cambridge.org/core/product/identifier/S1815240616000153/type/journal_article).
`main.py` displays a typical minimization process that returns a solution, and plots the result (in the exact same way as `nlc_plot.py`).

## Run

`nlc_plot.py` and `main.py` both accept command line arguments. One may run them with the `-h` flag to see detailed usage.

The `radial.json` configuration should return a *radial* state (spherical droplet, layers of biaxial discs surrounding a biaxial core), and the `tactoid.json` configuration should return a *tactoid* (elongated sphere, aligned uniaxial material pointing from one end to the other, almost no biaxiality).

Iteration converging to the tactoid is very slow. For more efficiency one may use the radial state (solved from `radial.json`) as the initial value.

## Output

Outputs are in the [VTK](https://vtk.org) format, a type of 3D data set that can be opened by [ParaView](https://paraview.org/download/).

To see the interiors of the biaxiality field, you may need to manually adjust the opacity.
