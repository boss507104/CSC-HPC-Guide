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

This configuration uses **uv** to resolve and install Python packages.

* **Fast Resolution** — uv resolves large scientific Python dependency trees substantially faster than conventional pip workflows.
* **Compatible Dependency Selection** — uv selects mutually compatible direct and transitive package versions.
* **Compiled Requirements** — The resolved package set is recorded in `requirements.txt`.
* **Consistent Workflow** — The same resolver handles dependency compilation and installation.

The direct package specifications in `requirements.in` intentionally avoid strict version pins. Each new dependency compilation may therefore resolve a newer compatible package set.

> [!NOTE]
> `requirements.in` records the requested top-level packages, while `requirements.txt` records the exact direct and transitive versions selected by uv. Rebuilding from an unchanged committed `requirements.txt` preserves the resolved package set, subject to platform and package availability.

> [!NOTE]
> Roihu does not provide `miniforge`, `conda`, or `mamba` modules. Dependency compilation therefore uses the available `python-data` module, a temporary Python virtual environment, and uv. The Tykky container build remains independent of this temporary resolver environment.

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
> Store the configuration files and temporary build data under your own `Utilities` directory on the parallel scratch filesystem.

---

## Dependency Overview

| Package | Version Policy | Purpose |
| --- | --- | --- |
| **Python** | 3.12 | Base interpreter supplied through the Tykky Conda specification |
| **python-data** | CSC-provided Python 3.12 environment | Runs the temporary uv resolver |
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

# Redirect temporary files and package caches to scratch
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
> Run this `srun` command only from a login node. If a compute node has already been allocated, continue in the current terminal and do not request another interactive allocation.

Reset the module environment and load the CSC Python 3.12 environment:

```bash
module purge
module load python-data
```

Create the temporary resolver environment:

```bash
rm -rf "$UV_RESOLVER_DIR"
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

Install uv inside the temporary resolver:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Compile `requirements.in` into `requirements.txt` for Python 3.12:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

The same command can be entered on one line:

```bash
uv pip compile "$PYTHON_ROOT/requirements.in" --output-file "$PYTHON_ROOT/requirements.txt" --python-version 3.12
```

> [!WARNING]
> A backslash must be the final character on its line. Do not insert a blank line after a line-ending backslash. Otherwise, Bash executes each following line as a separate command.

Incorrect:

```bash
uv pip compile \

    "$PYTHON_ROOT/requirements.in" \

    --output-file "$PYTHON_ROOT/requirements.txt"
```

Correct:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Inspect the beginning of the resolved file:

```bash
head -n 40 "$PYTHON_ROOT/requirements.txt"
```

Confirm that it is non-empty:

```bash
test -s "$PYTHON_ROOT/requirements.txt" && \
    echo "requirements.txt was generated successfully."
```

Deactivate and remove the temporary resolver:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

> [!NOTE]
> The first uv compilation may take time while uv downloads package metadata, resolves CUDA-related JAX packages, and fetches Git dependencies such as DataGraph.

> [!NOTE]
> Re-run this compilation step whenever packages are added to or removed from `requirements.in`.

> [!NOTE]
> The `python-data` module and temporary resolver environment are used only to produce `requirements.txt`. They are not included in the final Tykky environment.

---

## 3. Build the Tykky Container

Continue in the current compute-node session.

If no compute node is currently allocated, request one from a login node:

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

Reset the module environment and load Tykky:

```bash
module purge
module load tykky
```

Configure the temporary Tykky build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Verify that the required source files exist:

```bash
ls -l \
    "$PYTHON_ROOT/base4ML.yml" \
    "$PYTHON_ROOT/extra4ML.sh" \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/requirements.txt"
```

Confirm that `requirements.txt` is non-empty:

```bash
test -s "$PYTHON_ROOT/requirements.txt" || {
    echo "requirements.txt is missing or empty."
    exit 1
}
```

Remove an incomplete environment from a previous failed build:

```bash
rm -rf "$ENV_PREFIX"
```

Build the Tykky container:

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

Create the runtime initialisation script at `$BASE_SCRATCH/Python4ML.sh`:

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

Confirm the Python version:

```bash
python --version
```

> [!NOTE]
> `JAX_PLATFORMS="gpu"` requires a GPU allocation and compatible CUDA driver environment.

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

List the available kernels:

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

Reset the module environment and load `python-data`:

```bash
module purge
module load python-data
```

Create the temporary resolver:

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

Recompile the dependency set:

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

Inspect the changes:

```bash
git diff -- \
    "$PYTHON_ROOT/requirements.in" \
    "$PYTHON_ROOT/requirements.txt"
```

Rebuild or update the Tykky environment after confirming the resolved changes.

### Deliberately Refresh All Compatible Versions

Create and activate the temporary resolver as described above, then run:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12 \
    --upgrade
```

Deactivate and remove the resolver:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

> [!NOTE]
> Use `--upgrade` only when you intentionally want uv to refresh the compatible dependency versions.

### Repository Policy

For reproducible builds, commit both files:

```text
requirements.in
requirements.txt
```

Use `requirements.in` for reviewing direct dependency changes and `requirements.txt` for reconstructing the exact resolved environment.

---

## Adding or Updating Packages

Tykky updates should install the complete resolved dependency set rather than maintaining a separate incremental requirements file.

### 1. Edit the Direct Dependencies

```bash
nano -m "$PYTHON_ROOT/requirements.in"
```

### 2. Recompile the Dependency Set

Load the resolver environment:

```bash
module purge
module load python-data

rm -rf "$UV_RESOLVER_DIR"
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Compile the resolved dependencies:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

To deliberately refresh all compatible versions:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12 \
    --upgrade
```

Clean up the resolver:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

### 3. Create the Update Script

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

### 4. Apply the Update

Reset the module environment and load Tykky:

```bash
module purge
module load tykky
```

Configure the build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

mkdir -p "$TMPDIR"
```

Apply the update:

```bash
conda-containerize update \
    --post-install "$PYTHON_ROOT/update4ML.sh" \
    "$ENV_PREFIX"
```

> [!NOTE]
> A complete rebuild is safer after substantial changes to Python, JAX, CUDA, compilers, or binary dependencies.

---

## Rebuilding the Environment

Remove the current environment and temporary build directory:

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Confirm that the resolved requirements file exists:

```bash
test -s "$PYTHON_ROOT/requirements.txt" || {
    echo "requirements.txt is missing or empty."
    exit 1
}
```

Load Tykky:

```bash
module purge
module load tykky
```

Configure the build directory:

```bash
export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
```

Rebuild the environment:

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

Roihu may return:

```text
The following module(s) are unknown: "miniforge"
Unable to find: "conda"
Unable to find: "mamba"
```

This is expected and does not indicate a Tykky problem.

Use:

```bash
module purge
module load python-data
```

Then create a normal Python virtual environment:

```bash
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

### `/bin/activate: No Such File or Directory`

This means that `UV_RESOLVER_DIR` was empty when the activation command was executed.

Define it before creating the environment:

```bash
export UV_RESOLVER_DIR="$TMP_BUILD_DIR/uv-resolver"
```

Then recreate the resolver:

```bash
rm -rf "$UV_RESOLVER_DIR"
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"
```

### `uv pip compile` Reports That the Source File Is Missing

An error resembling the following:

```text
error: the following required arguments were not provided
bash: requirements.in: Permission denied
bash: --output-file: command not found
```

usually means that blank lines were inserted after line-ending backslashes.

Use the command without blank lines:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Alternatively, use one line:

```bash
uv pip compile "$PYTHON_ROOT/requirements.in" --output-file "$PYTHON_ROOT/requirements.txt" --python-version 3.12
```

### Dependency Resolution Appears to Pause on JAX CUDA Packages

Output such as:

```text
jax-cuda12-plugin
```

indicates that uv is resolving or downloading the JAX CUDA dependency set.

The initial resolution may take time because CUDA plugin wheels and package metadata are relatively large.

Do not interrupt the command unless it returns an explicit error.

### Total Environment Reset

```bash
rm -rf "$ENV_PREFIX"
rm -rf "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```

Then repeat dependency compilation and the Tykky build.

### `requirements.txt` Does Not Exist

Load the resolver environment:

```bash
module purge
module load python-data

rm -rf "$UV_RESOLVER_DIR"
python3 -m venv "$UV_RESOLVER_DIR"
source "$UV_RESOLVER_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install --no-cache-dir uv
```

Compile the file:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

Clean up:

```bash
deactivate
rm -rf "$UV_RESOLVER_DIR"
```

### Package Resolution Fails

Run:

```bash
uv pip compile \
    "$PYTHON_ROOT/requirements.in" \
    --output-file "$PYTHON_ROOT/requirements.txt" \
    --python-version 3.12
```

The resolver output should identify the incompatible direct dependencies.

### Tykky Is Not Available After Using `python-data`

The resolver environment and Tykky use separate module configurations.

Deactivate the resolver and reset the modules:

```bash
deactivate 2>/dev/null || true
module purge
module load tykky
```

### Package Installation Exceeds the Home Quota

The resolver and caches should remain under:

```text
$BASE_SCRATCH/.tykky_runtime
```

The relevant variables are:

```bash
echo "$TMP_BUILD_DIR"
echo "$UV_RESOLVER_DIR"
```

### JAX Reports That No GPU Is Available

GPU execution requires a GPU allocation.

For CPU-only use:

```bash
export JAX_PLATFORMS="cpu"
```

### Import Errors After an Incremental Update

Compare the installed environment with:

```bash
cat "$PYTHON_ROOT/requirements.txt"
```

When the environment becomes inconsistent, rebuild the complete Tykky image instead of applying additional incremental updates.

### Compiler Linkage Errors

Inspect the available compiler modules:

```bash
module avail gcc
module avail cmake
```

Load the compiler modules required by the affected package before starting the build.

### The Build Takes Too Long

Request a longer interactive allocation appropriate for the CSC system and partition.

Avoid running dependency compilation or environment builds directly on a login node.

---

## Notes

* The environment uses Python 3.12.
* `Harry` and `Dumbledore` are fictional placeholder values used in the public documentation.
* Replace `Harry` with the actual personal or shared directory under the CSC project.
* Replace `Dumbledore` with the preferred environment nickname.
* `PROJECT_USER_DIR` is not necessarily the same as the CSC login username.
* Roihu does not provide `miniforge`, `conda`, or `mamba` modules.
* Use `module load python-data` to create the temporary Python 3.12 resolver.
* Define `UV_RESOLVER_DIR` before creating or activating the resolver environment.
* The temporary resolver environment is separate from the final Tykky environment.
* Do not insert blank lines after line-ending backslashes in shell commands.
* `requirements.in` contains the direct dependency specifications.
* `requirements.txt` contains the exact direct and transitive versions resolved by uv.
* Commit both dependency files when reproducible builds matter.
* Recompile `requirements.txt` after changing `requirements.in`.
* Use `--upgrade` only when intentionally refreshing compatible package versions.
* Deactivate the resolver and run `module purge` before loading Tykky.
* `jax[cuda12]` installs the CUDA 12-compatible JAX package set, but GPU execution still requires a GPU allocation and compatible host drivers.
* Use batch or interactive compute nodes for dependency compilation, environment builds, and computational workloads.
* Avoid performing large package installations directly on CSC login nodes.
* Prefer a complete rebuild over repeated incremental updates when the dependency set changes substantially.
