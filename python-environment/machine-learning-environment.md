# ML Environment Configuration

Last updated: 3 July 2026

---

## Overview & Motivation

This folder contains configurations for deploying a reliable, high-performance runtime stack optimised for modern machine learning, statistics, and detailed chemical kinetics analysis on CSC supercomputers (**Puhti / Mahti / Roihu**). The setup focuses heavily on a high-throughput **JAX + Equinox + ONNX** development ecosystem.

Instead of deploying traditional Conda or pip environments directly on the parallel filesystem, we use **Tykky** to package the Python stack inside a single-file container image. This design reduces the Lustre parallel filesystem degradation caused by thousands of small metadata operations during Python package imports.

### Why Tykky?

* **Import Performance** — Library initialisation times drop from several minutes to seconds.
* **Reproducibility** — The complete execution stack remains packaged inside a single container image.
* **Startup Latency** — Fast environment startup proves valuable for high-volume, short MPI jobs.
* **Isolation** — The Python dependency stack remains separated from the cluster host environment.

### Why uv?

This configuration uses **uv** to resolve and install Python packages during the Tykky build.

* **Fast Resolution** — uv resolves large scientific Python dependency trees substantially faster than conventional pip workflows.
* **Compatible Dependency Selection** — uv selects mutually compatible direct and transitive package versions.
* **Compiled Requirements** — The resolved package set is recorded in `requirements.txt`.
* **Consistent Workflow** — The same resolver handles dependency compilation and installation.

The direct package specifications in `requirements.in` intentionally avoid strict version pins. Each new dependency compilation may therefore resolve a newer compatible package set.

> [!NOTE]
> `requirements.in` records the requested top-level packages, while `requirements.txt` records the exact direct and transitive versions selected by uv. Rebuilding from an unchanged committed `requirements.txt` preserves the resolved package set, subject to platform and package availability.

> [!NOTE]
> Roihu does not currently provide `miniforge`, `conda`, or `mamba` modules. Dependency compilation therefore uses the available `python/python` module, a temporary Python virtual environment, and uv. The Tykky container build itself remains independent of this temporary resolver environment.

---

## Global Configuration

Execute the following block to configure the project paths and environment name.

```bash
# --- USER CONFIGURATION START ---
export CSC_PROJECT="project_xxxxxxx"        # Your CSC project ID
export PROJECT_USER_DIR="Harry"             # Your directory under the CSC project
export ENV_NICKNAME="Dumbledore"            # Desired environment name
# --- USER CONFIGURATION END ---

# Derived paths
export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_ROOT="$BASE_SCRATCH/Python"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime"
export UV_RESOLVER_DIR="$TMP_BUILD_DIR/uv-resolver"

# Initialise directories
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "Configuration loaded for $CSC_PROJECT."
```

The configuration variables represent:

```text
CSC_PROJECT       CSC project ID
PROJECT_USER_DIR  Personal or shared directory under the CSC project
ENV_NICKNAME      Name assigned to the Python environment
```

For example:

```bash
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
```

The resulting base path is:

```text
/scratch/project_xxxxxxx/Harry/Utilities
```

> [!NOTE]
> `Harry` and `Dumbledore` are fictional example placeholder values used for public documentation. Replace them with your actual project directory and preferred environment name.

> [!NOTE]
> `PROJECT_USER_DIR` is not necessarily the same as your CSC login username. It identifies the directory located directly under the CSC project scratch path.

**Directory Structure**

```plaintext
/scratch/
└── $CSC_PROJECT/
    └── $PROJECT_USER_DIR/
        └── Utilities/                       # $BASE_SCRATCH
            ├── .tykky_runtime/              # $TMP_BUILD_DIR
            │   └── uv-resolver/             # $UV_RESOLVER_DIR
            └── Python/                      # $PYTHON_ROOT
                ├── base4ML.yml
                ├── extra4ML.sh
                ├── update4ML.sh
                ├── requirements.in
                ├── requirements.txt
                └── envs/
                    └── $ENV_NICKNAME-3.12/  # $ENV_PREFIX
```

> [!TIP]
> Store the configuration files and temporary build data under your own `Utilities` directory on the parallel scratch filesystem. Create the directory before starting the build.

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.12 | Base interpreter supplied through the Tykky Conda specification |
| **uv** | Latest available during dependency compilation and build | Python dependency resolution and installation |
| **NumPy** | Compatible version selected by uv | Core numerical array backend |
| **JAX** | Compatible CUDA 12 release selected by uv | Array programming and automatic differentiation |
| **Equinox** | Compatible version selected by uv | Neural-network and PyTree framework for JAX |
| **ONNX / jax2onnx** | Compatible versions selected by uv | Model export and interoperability |

The environment uses two dependency files:

```text
requirements.in   Direct, human-maintained package specifications
requirements.txt  Fully resolved direct and transitive package versions
```

---

## Installation Steps

### 1. Create the Configuration Files

Navigate to the Python configuration directory:

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 1.1 Create the Base Conda Specification

Create `base4ML.yml`:

```bash
nano -m base4ML.yml
```

Insert the following block:

```yaml
channels:
  - conda-forge
  - nodefaults

dependencies:
  - python=3.12
  - pip
  - git
  - compilers
  - cmake
  - make
  - ninja
```

### 1.2 Create the Direct Dependency Specification

Create `requirements.in`:

```bash
nano -m requirements.in
```

Insert the following block:

```text
# --- Core Math & Data ---
numpy
bottleneck
dask
h5py
pandas
polars
scipy
xarray
zarr

# --- Data Formats ---
netCDF4
pyarrow
pyfoam

# --- Data Acquisition ---
kagglehub

# --- JAX Ecosystem ---
jax[cuda12]
diffrax
distrax
einops
equinox
jax2onnx
jaxopt
jaxtyping
lineax
onnx
optax
optimistix
sympy2jax

# --- Machine Learning ---
catboost
feature-engine
gymnasium
lightgbm
linear-tree
mlflow
mlxtend
obliquetree
scikit-learn
tensorboard
wandb
xgboost

# --- Hyperparameter Optimisation ---
optuna

# --- Statistics ---
statsmodels

# --- Clustering & Dimensionality Reduction ---
hdbscan
igraph
leidenalg
umap-learn

# --- Physics & CFD ---
cantera
foamlib
meshio

# --- Mathematical Tools ---
numba
pint
ruptures
sympy
tensorly

# --- Custom Utilities ---
DataGraph @ git+https://github.com/boss507104/DataGraph.git#subdirectory=DataGraph

# --- Visualisation & UI ---
cmocean
colorcet
ipykernel
ipywidgets
IPython
ipyvtklink
k3d
matplotlib
plotly
pyvista
rich
scikit-image
seaborn
tqdm
trame
vtk

# --- Config & CLI ---
hydra-core
PyYAML

# --- HPC / Slurm ---
submitit

# --- System & Development ---
kneed
natsort
pytest
tabulate
typing-extensions
```

### 1.3 Create the Post-Installation Script

Create `extra4ML.sh`:

```bash
nano -m extra4ML.sh
```

Insert the following block:

```bash
#!/bin/bash
set -e

# Confirm that the build configuration is available
: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"

# Redirect temporary files and package caches to the scratch build directory
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Tykky build environment
python -m pip install --no-cache-dir uv

# Install the dependency set resolved in requirements.txt
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.txt"

# Remove package caches after installation
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x extra4ML.sh
```

---

## 2. Compile the Dependency Set

Request an interactive compute node before resolving the package set:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

> [!NOTE]
> If this command has already allocated a compute node, continue from the same terminal. Do not request a second interactive allocation from inside the first one.

Load the Python module available on Roihu:

```bash
module load python/python
```

Verify the available Python interpreter:

```bash
which python3
python3 --version
```

> [!NOTE]
> The Python interpreter used here only runs uv. The dependency target remains Python 3.12 because `uv pip compile` is called with `--python-version 3.12`.

Create a temporary virtual environment under the scratch build directory:

```bash
rm -rf "$UV_RESOLVER_DIR"

python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

Verify that the temporary environment is active:

```bash
which python
python --version
```

Upgrade pip and install uv:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Verify uv:

```bash
uv --version
```

Compile `requirements.in` into `requirements.txt` for Python 3.12:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Inspect the resolved file:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

Verify that the resolved file is non-empty:

```bash
test -s "$PYTHON_ROOT/requirements.txt" && \
    echo "requirements.txt was generated successfully."
```

Deactivate the temporary resolver environment:

```bash
deactivate
```

Remove the temporary resolver environment when it is no longer required:

```bash
rm -rf "$UV_RESOLVER_DIR"
```

> [!NOTE]
> Re-run this compilation step whenever you add, remove, or deliberately update packages in `requirements.in`.

> [!NOTE]
> The `python/python` module, temporary virtual environment, and uv installation are required only for compiling `requirements.txt`. They are not part of the final Tykky container.

---

## 3. Build the Tykky Container

Request an interactive compute node before running the container build if you are not already inside one:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

> [!NOTE]
> Skip this allocation command when continuing inside the compute-node session used to compile `requirements.txt`.

If package downloads or builds require more time, request a partition and time limit appropriate for the target CSC system.

Reset the module environment before loading Tykky:

```bash
module purge
module load tykky
```

Configure the temporary build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Verify that all required source files exist:

```bash
ls -l \
    "$PYTHON_ROOT/base4ML.yml" \
    "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/requirements.txt"
```

Verify that the resolved requirements file is non-empty:

```bash
test -s "$PYTHON_ROOT/requirements.txt" || {
    echo "requirements.txt is missing or empty."
    exit 1
}
```

Build the container:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

After a successful build, verify that the environment directory exists:

```bash
ls -ld "$ENV_PREFIX"
```

---

## Environment Activation / Loader

Create the runtime initialisation script at `$BASE_SCRATCH/Python4ML.sh`.

```bash
cat <<EOF > "$BASE_SCRATCH/Python4ML.sh"
#!/bin/bash

# Paths
export ENV_PREFIX="$ENV_PREFIX"

# Tykky container executable path
export PATH="\$ENV_PREFIX/bin:\$PATH"

# Prefer the JAX GPU backend when GPU resources are available
export JAX_PLATFORMS="gpu"
EOF
```

Make the loader executable:

```bash
chmod +x "$BASE_SCRATCH/Python4ML.sh"
```

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

Verify the active Python executable:

```bash
which python
python --version
```

> [!NOTE]
> `JAX_PLATFORMS="gpu"` requires a GPU allocation and compatible CUDA driver environment. Remove or override this variable when running CPU-only workloads.

For a CPU-only session:

```bash
export JAX_PLATFORMS="cpu"
```

---

## VS Code Kernel Registration

Register the Tykky Python environment as a Jupyter kernel for remote VS Code sessions.

Create the kernel directory:

```bash
mkdir -p "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml"
```

Create `kernel.json`:

```bash
cat <<EOF > "$HOME/.local/share/jupyter/kernels/$ENV_NICKNAME-ml/kernel.json"
{
  "argv": [
    "$ENV_PREFIX/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.12 ($ENV_NICKNAME Tykky ML)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF
```

Confirm the registration:

```bash
echo "Jupyter kernel '$ENV_NICKNAME-ml' has been registered."
```

List available kernels:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
jupyter kernelspec list
```

Remove an obsolete kernel when necessary:

```bash
jupyter kernelspec uninstall -f <kernel_name>
```

---

## Validation

Load the environment:

```bash
source "$BASE_SCRATCH/Python4ML.sh"
```

For CPU-only validation:

```bash
export JAX_PLATFORMS="cpu"
```

Verify the core package versions:

```bash
python -c "
import sys
import jax
import equinox as eqx
import numpy as np
from importlib.metadata import version

print(f'Python:     {sys.version.split()[0]}')
print(f'JAX:        {jax.__version__}')
print(f'Equinox:    {eqx.__version__}')
print(f'jax2onnx:   {version(\"jax2onnx\")}')
print(f'NumPy:      {np.__version__}')
print(f'Devices:    {jax.devices()}')
"
```

Verify that important scientific packages import correctly:

```bash
python -c "
import cantera
import h5py
import matplotlib
import onnx
import optax
import pandas
import scipy
import sklearn
import xarray

print('Core ML and scientific packages imported successfully.')
"
```

Inspect the installed package set:

```bash
python -m pip freeze
```

Compare it with the compiled requirements:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

---

## Dependency File Workflow

The dependency files serve different purposes:

```text
requirements.in
    Human-maintained list of direct dependencies.

requirements.txt
    uv-generated list containing exact direct and transitive versions.
```

### Add or Remove a Package

Edit `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Create the temporary uv resolver environment:

```bash
module load python/python

rm -rf "$UV_RESOLVER_DIR"

python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Recompile the dependency set:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Deactivate and remove the resolver environment:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

Inspect the changes:

```bash
git diff -- \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/requirements.txt"
```

Rebuild or update the Tykky environment after confirming the resolved changes.

### Deliberately Refresh All Compatible Versions

Create and activate the temporary resolver environment as described above, then run:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12 \
    --upgrade
```

Deactivate and remove the resolver environment:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

This command allows uv to select newer compatible package versions.

> [!NOTE]
> Without `--upgrade`, uv generally preserves existing compatible pins from the current `requirements.txt` when recompiling. Use `--upgrade` only when you intend to refresh the resolved dependency set.

### Repository Policy

For reproducible builds, commit both files:

```text
requirements.in
requirements.txt
```

Use `requirements.in` for reviewing direct dependency changes and `requirements.txt` for reconstructing the exact resolved environment.

---

## Adding or Updating Packages

Tykky updates should install the complete resolved dependency set rather than maintaining a separate update requirements file.

### 1. Edit the Direct Dependencies

Open `requirements.in`:

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

Add or remove the required packages.

### 2. Create the Temporary Resolver Environment

Load the available Python module:

```bash
module load python/python
```

Create and activate the temporary environment:

```bash
rm -rf "$UV_RESOLVER_DIR"

python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

Install uv:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

### 3. Recompile the Resolved Dependencies

Run:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

To deliberately refresh all compatible package versions:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12 \
    --upgrade
```

Deactivate and remove the resolver environment:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

### 4. Create the Update Script

Create `update4ML.sh`:

```bash
nano -m "$PYTHON_ROOT/update4ML.sh"
```

Insert the following block:

```bash
#!/bin/bash
set -e

# Confirm that the build configuration is available
: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"

# Redirect temporary files and package caches to scratch
export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install uv inside the active Tykky update environment
python -m pip install --no-cache-dir uv

# Apply the complete resolved dependency set
uv pip install \
    --requirements "$PYTHON_ROOT/requirements.txt"

# Remove package caches
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
```

Make the script executable:

```bash
chmod +x "$PYTHON_ROOT/update4ML.sh"
```

### 5. Apply the Update

Reset the module environment and load Tykky:

```bash
module purge
module load tykky
```

Configure the build directories:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Update the existing environment:

```bash
conda-containerize update \
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"
```

Group related dependency changes into one update to minimise repeated container repackaging.

> [!NOTE]
> Even when using a compiled `requirements.txt`, a full rebuild remains safer after substantial dependency, compiler, Python, CUDA, or binary-library changes.

---

## Rebuilding the Environment

Remove the current environment:

```bash
rm -rf "$ENV_PREFIX"
```

Clear the temporary build directory:

```bash
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Verify that the resolved dependency file exists:

```bash
ls -l "$PYTHON_ROOT/requirements.txt"
```

Verify that it is non-empty:

```bash
test -s "$PYTHON_ROOT/requirements.txt" || {
    echo "requirements.txt is missing or empty."
    exit 1
}
```

Reset the module environment and load Tykky:

```bash
module purge
module load tykky
```

Configure the Tykky build directories:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
```

Run the Tykky build again:

```bash
conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/base4ML.yml"
```

The rebuild uses the exact package versions recorded in `requirements.txt`.

---

## Troubleshooting

### `miniforge`, `conda`, or `mamba` Is Not Available

Roihu may return errors such as:

```text
The following module(s) are unknown: "miniforge"
Unable to find: "conda"
Unable to find: "mamba"
```

This is expected on Roihu and does not indicate a Tykky problem.

Use the available Python module instead:

```bash
module load python/python
```

Then create a standard virtual environment:

```bash
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

Install uv inside that environment:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

### Total Environment Reset

Remove the environment and temporary build directory:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Then repeat dependency compilation and the container build.

### `requirements.txt` Does Not Exist

Verify the direct dependency file:

```bash
ls -l "$PYTHON_ROOT/requirements.in"
```

Create the temporary resolver environment:

```bash
module load python/python

rm -rf "$UV_RESOLVER_DIR"

python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Compile the resolved requirements:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Deactivate and remove the resolver:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

### Python Virtual Environment Creation Fails

Verify the loaded Python module:

```bash
module list
which python3
python3 --version
```

Reload the Roihu Python module:

```bash
module purge
module load python/python
```

Retry:

```bash
python3 -m venv "$UV_RESOLVER_DIR"
```

If `python3` is unavailable but `python` exists, use:

```bash
python -m venv "$UV_RESOLVER_DIR"
```

### uv Cannot Be Installed

Inspect the temporary environment:

```bash
which python
python --version
python -m pip --version
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Retry the uv installation:

```bash
python -m pip install --no-cache-dir uv
```

### uv Resolves for the Wrong Python Version

The resolver environment Python version and target environment Python version do not need to be identical.

Always include:

```bash
--python-version 3.12
```

For example:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

The temporary Python environment runs uv, while `--python-version 3.12` defines the dependency-resolution target.

### Package Resolution Fails

Activate the temporary resolver environment and run:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

The resolver output should identify incompatible direct dependencies.

Because `requirements.in` does not enforce strict versions, remove or replace packages that impose mutually incompatible constraints.

### Package Installation Exceeds the Home Quota

Verify that the temporary directories point to scratch:

```bash
echo "$TMPDIR"
echo "$PIP_CACHE_DIR"
echo "$UV_CACHE_DIR"
echo "$UV_RESOLVER_DIR"
```

They should point under:

```text
$BASE_SCRATCH/.tykky_runtime
```

### Tykky Is Not Available After Loading the Python Module

The Python resolver environment and Tykky build environment use different module configurations.

Before building, reset the modules:

```bash
module purge
module load tykky
```

Verify Tykky:

```bash
which conda-containerize
conda-containerize --help
```

### JAX Reports That No GPU Is Available

Confirm that the shell runs inside a GPU allocation:

```bash
nvidia-smi
```

Check the JAX devices:

```bash
python -c "import jax; print(jax.devices())"
```

For CPU-only use:

```bash
export JAX_PLATFORMS="cpu"
```

### Import Errors After an Incremental Update

Inspect the installed versions:

```bash
python -m pip freeze
```

Compare them with:

```bash
cat "$PYTHON_ROOT/requirements.txt"
```

When the resulting environment becomes inconsistent, rebuild the complete Tykky image rather than stacking further updates.

### Compiler Linkage Errors

Inspect the available compiler modules:

```bash
module avail gcc
module avail cmake
```

Load the compiler modules required by the affected package before starting the Tykky build.

### The Build Takes Too Long

Request a longer interactive allocation appropriate for the CSC system and partition.

Avoid running environment builds directly on a login node.

---

## Notes

* The environment uses Python 3.12.
* `Harry` and `Dumbledore` are fictional placeholder values used in the public documentation.
* Replace `Harry` with the actual personal or shared directory under the CSC project.
* Replace `Dumbledore` with the preferred environment nickname.
* `PROJECT_USER_DIR` identifies the personal or shared directory directly under the CSC project scratch path.
* `PROJECT_USER_DIR` is not necessarily the same as the CSC login username.
* Roihu does not provide `miniforge`, `conda`, or `mamba` modules.
* Use `module load python/python` and a temporary virtual environment to run uv.
* The temporary resolver environment is separate from the final Tykky environment.
* `requirements.in` contains the direct dependency specifications.
* `requirements.txt` contains the exact direct and transitive versions resolved by uv.
* Commit both dependency files when reproducible builds matter.
* Recompile `requirements.txt` after changing `requirements.in`.
* Use `--upgrade` only when intentionally refreshing compatible package versions.
* Reset the module environment with `module purge` before loading Tykky after using the Python resolver module.
* `jax[cuda12]` installs the CUDA 12-compatible JAX package set, but GPU execution still requires a GPU allocation and compatible host drivers.
* The Tykky image should be treated as the deployed runtime unit.
* Use batch or interactive compute nodes for dependency compilation, environment builds, and computational workloads.
* Avoid performing large package installations directly on CSC login nodes.
* Prefer a complete rebuild over repeated incremental updates when the dependency set changes substantially.
