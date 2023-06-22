# FWI-LS Project

Solver for Full Waveform Inversion (FWI) using Level Set in the frequency domain.
This software is part of a project for my masters' degree in Mechanical Engineering (UFSC, 2019) -
[The Dissertation is available here](https://repositorio.ufsc.br/handle/123456789/236869).

This software is implemented using the Finite Element Method (FEM) with 2d linear quadrilateral elements for spatial
discretization. Parallelization on the frequency domain, with options to parallel linear solver configurations
and parallel shots (per source).

Project uses `conda` to manage it's Python dependencies. `cmake` and a C++ compiler are also necessary in order to
compile the C++ code used to improve Python performance in specific contexts.
The C++ bindings are compiled and placed on `src/cpp/build`. This path must be on `PYTHONPATH`.


## Level Set Evolution Algorithm

A Level Set function is used to approximate the solution. For each optimization step, the Level Set evolution direction is
calculated using an "evolution velocity", which is determined using the Diffusion-Reaction equation, which depends on
information from the sensibility analysis. The step, on the other hand, must be guessed by the user. Based on this
guess, the algorithm will then search for a more suitable step, to guarantee that the objective function from the
optimization procedure is decreased.


## Useful tools

- `inv compile` Compiles the project using cmake and gcc
- `inv clean` Cleanup compilation artifacts
- `inv run-fwi-case` Runs FWI case from database of cases. Options:
  - `--case-name` (required) which case to run
  - `--iterations` Number of iterations to run (default=10)
  - `--threads` Number of parallel shots (default=1)
  - `--no-plot` Disable output plots
  - `--ignore-LS-V-at-sponge` If set, will not generate inclusions in the sponge area (When using sponge absorbing layers)
  - `--min-obj-function-ratio-change` Set the minimum change between optimization steps (Values lower than this will assume the optimization has converged)
  - Level Set Evolution options (See Level Set Evolution section on this file for more information)
    - `initial-evolution-step` Initial guess for dtheta on the LS evolution algorithm
    - `min-evolution-step` Minimum step size for dtheta on the LS evolution
- `inv run-fwd-case` Runs forward case from database of cases. Options:
  - `--case-name` (required) which case to run
  - `--omega` Frequency to run, in rad/s (default=5.0)
  - `--size` Mesh size (size == nx == ny) (default=80)
  - `--threads` Number of threads for the linear solver (default=1)

