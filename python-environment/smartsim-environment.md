# SmartSim Environment Configuration

Last updated: 21 July 2026

> [!TIP]
> ## One-Command Installation
>
> Copy `smartsim-python.sh` to `/scratch/project_xxxxxxx/PROJECT_USER_DIR/`, then run:
>
> ```bash
> chmod +x smartsim-python.sh
> ./smartsim-python.sh
> ```

---

## Overview & Motivation

This folder deploys a **unified ML + SmartSim/SmartRedis stack** on CSC supercomputers (**Puhti / Mahti / Roihu**) — **JAX + Equinox + TensorFlow + PyTorch + ONNX + PySR (JuliaCall) + SmartSim + SmartRedis**, all in one environment.

This environment is now a **superset of the previously separate `PythonML` stack**. Every package and workflow that used to live in the standalone ML environment — including PySR's Julia toolchain — is included here, built and validated on both `x64` and `arm64`. The earlier guidance to keep the ML and SmartSim stacks strictly separate no longer applies once PySR/Julia support is folded in this way; a standalone `PythonML/` environment is not needed if you use this stack.

SmartSim and SmartRedis install directly from the CSC-maintained forks (`csc-develop` branch), which already contain:

* **Python 3.12 support**
* **NumPy 2.x compatibility**
* **Linux ARM64 platform support** — upstream SmartSim only shipped a `Darwin+ARM64+CPU` platform config; the fork adds `Linux+ARM64+CPU` and the missing `aarch64`→`arm64` architecture-string mapping
* **RedisAI TensorFlow backend on Linux ARM64**
* **RedisAI ONNX Runtime backend on Linux ARM64**
* **RedisAI LibTorch backend on Linux ARM64**
* **SmartRedis compiler/source fixes**

**No post-install source patching is required.** `smart build` runs identically on `x64` and `arm64`, building the RedisAI TensorFlow, ONNX Runtime, and LibTorch backends automatically.

Separately, **PySR's Julia dependency is resolved and precompiled at build time**, exactly as it was in the standalone ML stack: `juliapkg` fetches Julia and the `PythonCall`/`SymbolicRegression` packages inside the Tykky build. Immediately after the Tykky build succeeds, the installer copies that packaged Julia *project* **once** into a writable scratch location (the Tykky image itself is read-only and `juliapkg` needs to write a lock file there); the Julia *depot* directory is created alongside it. Sourcing the loader afterwards only points environment variables at these already-prepared directories — it no longer copies or deletes anything. `PYTHON_JULIAPKG_OFFLINE=yes` at runtime prevents any accidental re-download.

RedisAI model execution is available — TensorFlow, ONNX, and PyTorch (via LibTorch) models can be executed with `set_model`/`run_model` — but the primary workflow may still run JAX/Equinox/PySR inference in external Python workers. SmartRedis carries tensors, weights, metrics, and predictions either way.

We use **Tykky** to package the whole Python stack into a single-file container image, avoiding Lustre metadata slowdowns from thousands of small file imports.

A Tykky container built for one architecture will not run on the other. The **SmartRedis native library** (Section 6, used for OpenFOAM/C++/Fortran linkage) is built separately per architecture via CMake, independent of the `smart build` step described above.

**Why Tykky:** near-instant imports, a single reproducible image, fast startup, isolation from the host environment.

**Why uv:** fast resolution/installation, plus `uv pip check` to validate the final dependency graph. `--link-mode=copy` is used throughout since the uv cache and the Tykky build environment live on different filesystems.

```text
Python        3.12
SmartSim      0.8.0 (CSC fork: PentagonToy/SmartSim @ csc-develop)
SmartRedis    0.6.1-compatible (CSC fork: PentagonToy/SmartRedis @ csc-develop)
JAX           resolved at build time (CUDA 12 on arm64)
TensorFlow    2.18.1
PyTorch       2.7.1
ONNX          resolved (+ ONNX Runtime, tf2onnx, skl2onnx)
PySR / Julia  resolved + precompiled at build time (JuliaCall); writable runtime copy prepared once, at build time
NumPy         >= 2.0
protobuf      resolved by uv (no longer hard-pinned)
CMake         resolved (no longer pinned < 3.30.0)
RedisAI       TensorFlow + ONNX Runtime + LibTorch backends, built on both x64 and arm64
```

```text
requirements.in            Human-maintained direct package specifications and compatibility constraints
requirements-$ENV_ARCH.txt Installed-state snapshot recorded after a successful build (excludes SmartSim/SmartRedis)
julia-environment-$ENV_ARCH.txt  Julia toolchain + package status recorded after a successful build
runtime-$ENV_ARCH.sh       GCC module recorded at build time, read by the loader on every source
```

Build order: install `uv` → install the full `requirements.in` set (including `pysr`/`julia`, TensorFlow, PyTorch, ONNX) → resolve/precompile Julia+PySR → install SmartRedis (fork) → install SmartSim (fork) → `smart build` (Redis + RedisAI, all backends) → restore `requirements.in` → `uv pip check` → **prepare the writable Julia runtime once** → build the native SmartRedis library and record its GCC module.

Part of the [CSC Environment Helpers Framework](https://github.com/PentagonToy/CSCEnvironmentHelpers). Production examples live in [SmartSim4CSC](https://github.com/PentagonToy/SmartSim4CSC).

---

## Build Flow

```text
Set identity once (Section 0)
  |
  v
Choose target architecture
  |
  +-- x64  (Roihu CPU / Puhti / Mahti)
  |     Global Config (x64) --> install full requirements.in
  |     --> resolve/precompile Julia + PySR
  |     --> install SmartRedis + SmartSim (csc-develop fork)
  |     --> build Tykky env (Redis + RedisAI, all backends, via smart build)
  |     --> prepare writable Julia runtime ONCE (copy julia_env, create depot dir)
  |     --> build SmartRedis-x64 native library; record GCC module used
  |
  +-- arm64 (Roihu GPU)
        Global Config (arm64) --> install full requirements.in
        --> resolve/precompile Julia + PySR
        --> install SmartRedis + SmartSim (csc-develop fork)
        --> build Tykky env (Redis + RedisAI, all backends — same as x64)
        --> prepare writable Julia runtime ONCE (copy julia_env, create depot dir)
        --> build SmartRedis-arm64 native library; record GCC module used
        --> also runs JAX/Equinox/TensorFlow/PyTorch/PySR training and inference locally

After the required track(s) are built:
  Create Python4SmartSim.sh --> source Python4SmartSim.sh
  --> loader picks x64/arm64 from `uname -m`, only sets environment variables
      and PATH/LD_LIBRARY_PATH/CMAKE_PREFIX_PATH (idempotent — safe to re-source)
  --> Jupyter kernels run through a launcher wrapper that sources the same loader
```

Skip the `arm64` track entirely if you never run workloads on Roihu GPU nodes against this stack.

---

## 0. One-Time Identity Configuration

Every script needs three values: your CSC project ID, your directory under that project, and the environment nickname. Set them **once** in a file under `$HOME`.

> `Harry`, `Dumbledore`, and `project_xxxxxxx` are fictional placeholders. Fill in real values **only here**.

```bash
mkdir -p "$HOME/.config/csc-hpc"

cat <<'EOF' > "$HOME/.config/csc-hpc/identity.sh"
export CSC_PROJECT="project_xxxxxxx"
export PROJECT_USER_DIR="Harry"
export ENV_NICKNAME="Dumbledore"
EOF

chmod 600 "$HOME/.config/csc-hpc/identity.sh"
```

Edit with real values, then verify:

```bash
nano "$HOME/.config/csc-hpc/identity.sh"

source "$HOME/.config/csc-hpc/identity.sh"
echo "CSC_PROJECT=$CSC_PROJECT"
echo "PROJECT_USER_DIR=$PROJECT_USER_DIR"
echo "ENV_NICKNAME=$ENV_NICKNAME"
```

`ENV_ARCH` is **not** part of this file — it's chosen per build in Section 1, and auto-detected via `uname -m` in the loader / `smartsim-update`.

---

## 1. Global Configuration

Run **one** block per node. Only `ENV_ARCH` differs.

### 1.1 x64 (Roihu CPU / Puhti / Mahti)

```bash
source "$HOME/.config/csc-hpc/identity.sh"
export ENV_ARCH="x64"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

### 1.2 arm64 (Roihu GPU)

```bash
source "$HOME/.config/csc-hpc/identity.sh"
export ENV_ARCH="arm64"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"
export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"

mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"

echo "ENV_ARCH=$ENV_ARCH"
echo "PYTHON_ROOT=$PYTHON_ROOT"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
echo "TMP_BUILD_DIR=$TMP_BUILD_DIR"
```

**Directory layout:**

```text
/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities/          # $BASE_SCRATCH
├── .tykky_runtime_smartsim_x64/
├── .tykky_runtime_smartsim_arm64/
├── .julia_env_runtime_x64/       # created ONCE, right after the Tykky build (Section 5.1)
├── .julia_env_runtime_arm64/     # created ONCE, right after the Tykky build (Section 5.1)
├── .julia_depot_runtime_x64/     # created ONCE, right after the Tykky build (Section 5.1)
├── .julia_depot_runtime_arm64/   # created ONCE, right after the Tykky build (Section 5.1)
├── Python4SmartSim.sh
├── SmartRedis-x64/                                  # $SMARTREDIS_DIR (x64)
├── SmartRedis-arm64/                                # $SMARTREDIS_DIR (arm64)
└── Python/                                           # $PYTHON_BASE
    └── PythonSmartSim/                               # $PYTHON_ROOT
        ├── base4SmartSim.yml
        ├── extra4SmartSim.sh
        ├── update4SmartSim.sh
        ├── requirements.in
        ├── requirements-x64.txt
        ├── requirements-arm64.txt
        ├── julia-environment-x64.txt
        ├── julia-environment-arm64.txt
        ├── runtime-x64.sh          # records the GCC module used for the x64 native build
        ├── runtime-arm64.sh        # records the GCC module used for the arm64 native build
        ├── jupyter-kernel-x64.sh   # kernel launcher wrapper (Section 8)
        ├── jupyter-kernel-arm64.sh # kernel launcher wrapper (Section 8)
        └── envs/
```

The `.julia_env_runtime_*` / `.julia_depot_runtime_*` directories and `runtime-$ENV_ARCH.sh` are created **once**, at build time (Sections 5.1 and 6) — not by the loader. Sourcing `Python4SmartSim.sh` only ever reads them; it never copies, deletes, or recreates them. Re-run the installer for a given architecture if these directories are missing or out of date.

---

## 2. Dependency Overview

| Package | Version | Purpose |
| --- | --- | --- |
| Python | 3.12 | Base interpreter |
| uv | latest at build | Resolution, installation, `uv pip check` |
| SmartSim | `PentagonToy/SmartSim @ csc-develop` | Orchestration; Redis + RedisAI lifecycle on both architectures |
| SmartRedis | `PentagonToy/SmartRedis @ csc-develop` | Python client + native C++/Fortran library, both architectures |
| JAX / Equinox / distrax / distreqx | resolved at build time; CUDA 12 on arm64 | Autodiff / training / inference / probabilistic modelling |
| TensorFlow | 2.18.1 | Python framework + source for the RedisAI TensorFlow backend |
| PyTorch | 2.7.1 | Python framework; RedisAI executes via the LibTorch backend |
| ONNX / ONNX Runtime | resolved at build time | Model interchange + Python-side ONNX Runtime |
| PySR / julia (JuliaCall) | resolved | Symbolic regression; Julia toolchain resolved and precompiled at build time, then copied into a writable runtime location once |
| shap | resolved | Model explainability |
| dvc | resolved | Data version control |
| nbconvert / papermill | resolved | Notebook execution and export |
| optuna / optuna-dashboard | resolved | Hyperparameter optimisation + web UI |
| NumPy | `>= 2.0` | No longer pinned below 2.0 |
| pydantic / loguru / pyinstrument | resolved | Config validation, structured logging, profiling |
| RedisAI backends | TensorFlow + ONNX Runtime + LibTorch, built on **both** architectures | Fetched automatically from GitHub Releases during `smart build` |

---

## 3. Create the Configuration Files

```bash
mkdir -p "$PYTHON_ROOT"
cd "$PYTHON_ROOT"
```

### 3.1 `base4SmartSim.yml`

```bash
cat <<'EOF' > "$PYTHON_ROOT/base4SmartSim.yml"
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
EOF
```

### 3.2 `requirements.in`

Do **not** add `smartsim` or `smartredis` here — both install separately from the fork in `extra4SmartSim.sh`.

```bash
cat <<'EOF' > "$PYTHON_ROOT/requirements.in"
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
distreqx
equinox
jaxtyping
jax2onnx
jaxopt
einops
lineax
optax
optimistix
sympy2jax

# --- TensorFlow / PyTorch / ONNX ---
tensorflow==2.18.1
torch==2.7.1
onnx
onnxruntime
tf2onnx
skl2onnx

# --- Machine Learning ---
catboost
feature-engine
gymnasium
lightgbm
linear-tree
mlflow
mlxtend
scikit-learn
shap
tensorboard
treeple
wandb
xgboost

# --- Symbolic Regression & Julia ---
pysr
julia

# --- Hyperparameter Optimisation ---
optuna
optuna-dashboard

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

# --- Data Version Control ---
dvc

# --- Custom Utilities ---
DataGraph @ git+https://github.com/PentagonToy/DataGraph.git#subdirectory=DataGraph
eqx_io @ git+https://github.com/PentagonToy/CSC-HPC-Guide.git#subdirectory=utilities/eqx4smartredis

# --- Notebook Execution ---
ipykernel
ipywidgets
IPython
nbconvert
papermill

# --- Visualisation & UI ---
cmocean
colorcet
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
pydantic
PyYAML

# --- Profiling & Logging ---
loguru
pyinstrument

# --- HPC / Slurm ---
submitit

# --- System & Development ---
kneed
natsort
pytest
tabulate
typing-extensions
EOF
```

`tensorflow==2.18.1` and `torch==2.7.1` remain pinned. `jax[cuda12]`, ONNX, ONNX Runtime, `protobuf`, and `numpy` are deliberately left unpinned and resolve to the newest compatible versions at build time. Python-side framework versions do not need to exactly match the RedisAI backend versions built by `smart build`, since they're separate runtime components — validate exported models against the corresponding RedisAI backend (Section 13) rather than assuming version equality. `pysr`/`julia` require the resolve/precompile step below — adding them to `requirements.in` alone is not sufficient.

### 3.3 `extra4SmartSim.sh` (post-install, runs *inside* the build)

This installs the full package set, resolves and precompiles PySR's Julia dependency (identical to the standalone ML stack's process), then installs and builds SmartSim/SmartRedis with all RedisAI backends. `smart build` runs identically on both architectures, with no runtime patching.

```bash
cat <<'EOF' > "$PYTHON_ROOT/extra4SmartSim.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"
: "${ENV_ARCH:?ENV_ARCH is not set}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

# --- Resolve and precompile PySR's Julia dependency ---
# Always derive the Julia paths from the *actual* Python sys.prefix inside
# the container, not from $ENV_PREFIX — Tykky's wrapper path and the real
# in-container prefix are not the same thing.
PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"
mkdir -p "$JULIA_DEPOT_PATH" "$PYTHON_JULIAPKG_PROJECT"

python - <<'PY'
import juliapkg
juliapkg.resolve()
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia project:    {juliapkg.project()}")
PY

python - <<'PY'
import pysr
print(f"PySR version: {pysr.__version__}")
PY

python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using Pkg; Pkg.instantiate(); Pkg.precompile(); "
     "using PythonCall; using SymbolicRegression"],
    check=True,
)
PY

# --- SmartRedis + SmartSim, from the CSC-maintained forks (both architectures) ---
# csc-develop already contains: the SmartRedis <cstdint> compiler fix, the
# SmartSim Linux+ARM64+CPU platform config, and the aarch64->arm64
# architecture-string mapping. No post-install patching is needed.
uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@csc-develop"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@csc-develop"

# --- Build the Orchestrator (Redis + RedisAI backends) — both architectures ---
export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages

# Restore packages potentially disturbed by the build above
uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

# Record installed versions; SmartSim/SmartRedis excluded (installed
# fresh from source every build, never replayed).
python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

python - <<'PY' > "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
print(f"Julia executable: {julia}")
print(f"Julia project: {project}\n")
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using InteractiveUtils; versioninfo(); using Pkg; Pkg.status()"],
    check=True,
)
PY

rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
```

If `smart build` reports incompatible-pointer-type compile errors on some GCC versions, retry with `CFLAGS="-Wno-incompatible-pointer-types" CXXFLAGS="-Wno-incompatible-pointer-types"` prefixed to the `smart clobber`/`smart build` lines — see Section 13.

---

## 4. Request a Build Node

> **Tip — downloads:** the SmartRedis/SmartSim fork installs, `smart build`'s automatic backend downloads (TensorFlow, ONNX Runtime, LibTorch from GitHub Releases), and Julia's package resolution all need outbound internet access. If a compute allocation's network is restricted, try the download-heavy steps on the login node first.

**x64:**

```bash
srun --account="$CSC_PROJECT" \
    --partition=small \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=01:30:00 \
    --pty bash
```

**arm64 (Roihu GPU):**

```bash
sinteractive \
    --account "$CSC_PROJECT" \
    --gpu \
    --cores 36 \
    --time 01:30:00
```

If Section 1's variables aren't inherited, re-run the matching Global Configuration block.

---

## 5. Build the Tykky Environment

```bash
module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"

ls -l \
    "$PYTHON_ROOT/base4SmartSim.yml" \
    "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/requirements.in"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize new \
    --prefix "$ENV_PREFIX" \
    --post-install "$PYTHON_ROOT/extra4SmartSim.sh" \
    "$PYTHON_ROOT/base4SmartSim.yml"
```

Check the result:

```bash
ls -ld "$ENV_PREFIX"
ls -lh "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"
ls -lh "$PYTHON_ROOT/julia-environment-$ENV_ARCH.txt"

"$ENV_PREFIX/bin/python" -m pip list --format=freeze \
    | grep -E '^(jax|numpy|tensorflow|torch|onnx|onnxruntime|pysr|smartsim|smartredis)=='
```

### 5.1 Prepare the writable PySR / Julia runtime (once)

The Tykky image is read-only, but `juliapkg` needs to write a lock file into the Julia project directory the first time it's used. Rather than doing this copy on every `source` (as in earlier versions of this guide), it now happens **once**, right after the Tykky build succeeds:

```bash
echo "Preparing writable PySR / Julia runtime..."

PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"
JULIA_ENV_SOURCE="$PYTHON_PREFIX/julia_env"
JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

if [ ! -d "$JULIA_ENV_SOURCE" ]; then
    echo "ERROR: Packaged Julia environment was not found: $JULIA_ENV_SOURCE"
    exit 1
fi

rm -rf "$JULIA_ENV_RUNTIME"
cp -a "$JULIA_ENV_SOURCE" "$JULIA_ENV_RUNTIME"
mkdir -p "$JULIA_DEPOT_RUNTIME"

echo "Julia environment: $JULIA_ENV_RUNTIME"
echo "Julia depot:       $JULIA_DEPOT_RUNTIME"
```

Rerun this block (and only this block) if the Tykky environment is rebuilt for this architecture — the packaged Julia project may have changed.

Build the other architecture separately (Section 1 + Section 4).

---

## 6. Build the SmartRedis Native Library

Needed on **both** architectures for OpenFOAM/C++/Fortran linkage — this is a separate CMake build, unrelated to `smart build`/RedisAI or the Julia toolchain.

Request a node (Section 4), then set the GCC module for your target system and load compilers:

```bash
module purge
```

Compilers, e.g. Roihu CPU:

```bash
# Roihu CPU
export GCC_MODULE="gcc/13.4.0"
module load "$GCC_MODULE"
module load cmake/3.26.5
```

Roihu GPU:

```bash
# Roihu GPU
export GCC_MODULE="gcc/13.4.0"
module load "$GCC_MODULE"
module load cmake/3.31.11
```

or Mahti:

```bash
export GCC_MODULE="gcc/13.1.0"
module load "$GCC_MODULE"
module load cmake/3.28.6
module load git
```

Record the GCC module for the loader so it never has to guess based on hostname:

```bash
cat <<EOF > "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
export SMARTSIM_GCC_MODULE="$GCC_MODULE"
EOF
chmod 600 "$PYTHON_ROOT/runtime-$ENV_ARCH.sh"
```

Clone and build:

```bash
cd "$BASE_SCRATCH"
rm -rf "$SMARTREDIS_DIR"

git clone \
    --branch csc-develop \
    https://github.com/PentagonToy/SmartRedis.git \
    "$SMARTREDIS_DIR"

cd "$SMARTREDIS_DIR"
rm -rf build install
```

```bash
env \
    -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u CC -u CXX -u FC \
    CC=gcc CXX=g++ FC=gfortran \
    make lib-with-fortran
```

Verify:

```bash
find "$SMARTREDIS_DIR/install" -maxdepth 3 -type f | sort

# If lib64 doesn't exist, use lib.
ls -la "$SMARTREDIS_DIR/install/lib64"
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library installed successfully."
ldd "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so"
```

---

## 7. Loader — `Python4SmartSim.sh`

The loader is now a **pure environment loader** — it must be *sourced*, not executed, and re-sourcing it repeatedly in the same shell is safe. It:

* validates that the Tykky environment, SmartRedis install, and the writable Julia runtime directories (created once in Sections 5.1/6) all exist;
* reads `runtime-$ENV_ARCH.sh` to learn which GCC module the native SmartRedis library was built against, and loads it only if not already loaded (`module is-loaded`);
* prepends to `PATH`, `LD_LIBRARY_PATH`, and `CMAKE_PREFIX_PATH` through a small `path_prepend` helper that skips directories already present, so re-sourcing never creates duplicate entries;
* never copies, deletes, or installs anything.

```bash
cat <<'EOF' > "$BASE_SCRATCH/Python4SmartSim.sh"
#!/bin/bash
#
# SmartSim Python environment loader
#
# Usage:
#   source /scratch/<project>/<user>/Utilities/Python4SmartSim.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This file must be sourced, not executed:"
    echo "    source ${BASH_SOURCE[0]}"
    exit 1
fi

IDENTITY_FILE="$HOME/.config/csc-hpc/identity.sh"

if [ ! -f "$IDENTITY_FILE" ]; then
    echo "Identity file not found: $IDENTITY_FILE"
    return 1
fi

source "$IDENTITY_FILE"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"

case "$(uname -m)" in
    x86_64)
        export ENV_ARCH="x64"
        export KERNEL_ARCH="x86_64"
        export JAX_PLATFORMS="cpu"
        ;;
    aarch64)
        export ENV_ARCH="arm64"
        export KERNEL_ARCH="aarch64"
        export JAX_PLATFORMS="cuda"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        return 1
        ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export SMARTREDIS_DIR="$BASE_SCRATCH/SmartRedis-$ENV_ARCH"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Python environment not found:"
    echo "    $ENV_PREFIX"
    return 1
fi

if [ ! -d "$SMARTREDIS_DIR/install" ]; then
    echo "SmartRedis installation not found:"
    echo "    $SMARTREDIS_DIR/install"
    return 1
fi

RUNTIME_CONFIG="$PYTHON_ROOT/runtime-$ENV_ARCH.sh"

if [ -f "$RUNTIME_CONFIG" ]; then
    source "$RUNTIME_CONFIG"
fi

if [ -n "${SMARTSIM_GCC_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module is-loaded "$SMARTSIM_GCC_MODULE" 2>/dev/null ||
        module load "$SMARTSIM_GCC_MODULE"
fi

path_prepend() {
    local variable_name="$1"
    local directory="$2"
    local current_value="${!variable_name-}"

    case ":$current_value:" in
        *":$directory:"*)
            ;;
        *)
            printf -v "$variable_name" '%s' \
                "$directory${current_value:+:$current_value}"
            export "$variable_name"
            ;;
    esac
}

if [ -d "$SMARTREDIS_DIR/install/lib64" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib64"
elif [ -d "$SMARTREDIS_DIR/install/lib" ]; then
    export SMARTREDIS_LIB_DIR="$SMARTREDIS_DIR/install/lib"
else
    echo "SmartRedis library directory not found."
    return 1
fi

path_prepend PATH "$ENV_PREFIX/bin"
path_prepend LD_LIBRARY_PATH "$SMARTREDIS_LIB_DIR"
path_prepend CMAKE_PREFIX_PATH "$SMARTREDIS_DIR/install"

export SMARTSIM_DB_FILE_PARSE_TRIALS=600

# PySR / Julia runtime paths — prepared ONCE at build time (Sections 5.1/6);
# this loader only points environment variables at them.
export PYTHON_PREFIX="$("$ENV_PREFIX/bin/python" -c 'import sys; print(sys.prefix)')"
export JULIA_ENV_RUNTIME="$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH"
export JULIA_DEPOT_RUNTIME="$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"

if [ ! -d "$JULIA_ENV_RUNTIME" ]; then
    echo "Writable Julia environment not found:"
    echo "    $JULIA_ENV_RUNTIME"
    echo "Run the SmartSim installer again for $ENV_ARCH."
    return 1
fi

mkdir -p "$JULIA_DEPOT_RUNTIME"

export PYTHON_JULIAPKG_PROJECT="$JULIA_ENV_RUNTIME"
export JULIA_DEPOT_PATH="$JULIA_DEPOT_RUNTIME:$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_OFFLINE="yes"
export PYTHON_JULIACALL_THREADS="${SLURM_CPUS_PER_TASK:-auto}"

unset PYTHON_JULIACALL_EXE
unset PYTHON_JULIACALL_PROJECT

export JUPYTER_KERNEL_NAME="$ENV_NICKNAME-smartsim-$KERNEL_ARCH"
export JUPYTER_KERNEL_DISPLAY="Python 3.12 ($ENV_NICKNAME SmartSim $KERNEL_ARCH)"
export JUPYTER_KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$JUPYTER_KERNEL_NAME"

if [ "${SMARTSIM_ENV_QUIET:-0}" != "1" ]; then
    echo "SmartSim Python environment loaded"
    echo "ENV_ARCH=$ENV_ARCH"
    echo "ENV_PREFIX=$ENV_PREFIX"
    echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"
    echo "JAX_PLATFORMS=$JAX_PLATFORMS"
    echo "PYTHON_JULIAPKG_PROJECT=$PYTHON_JULIAPKG_PROJECT"
fi

unset -f path_prepend
EOF
chmod +x "$BASE_SCRATCH/Python4SmartSim.sh"
```

Load it:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
echo "$PYTHON_ROOT"; echo "$ENV_PREFIX"; echo "$SMARTREDIS_DIR"
python --version
```

Confirm re-sourcing doesn't duplicate `PATH`:

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"
source "$BASE_SCRATCH/Python4SmartSim.sh"
echo "$PATH" | tr ':' '\n' | grep PythonSmartSim
```

The environment path should appear exactly once.

`SMARTSIM_ENV_QUIET=1` suppresses the status banner — used internally by the Jupyter kernel launcher (Section 8).

---

## 8. Register the Jupyter Kernel

Run once per architecture. Rather than baking `ENV_PREFIX` and Julia/JAX environment variables directly into `kernel.json`, the kernel now runs through a small launcher wrapper that sources the exact same loader used interactively — so a notebook kernel always matches a terminal session, including whatever the loader does at that moment.

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

# --- Kernel launcher wrapper ---
JUPYTER_KERNEL_LAUNCHER="$PYTHON_ROOT/jupyter-kernel-$ENV_ARCH.sh"

cat <<EOF > "$JUPYTER_KERNEL_LAUNCHER"
#!/bin/bash
export SMARTSIM_ENV_QUIET=1
source "$BASE_SCRATCH/Python4SmartSim.sh" || exit 1
unset SMARTSIM_ENV_QUIET
exec "$ENV_PREFIX/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$JUPYTER_KERNEL_LAUNCHER"

# --- kernel.json pointing at the wrapper ---
mkdir -p "$JUPYTER_KERNEL_DIR"

cat <<EOF > "$JUPYTER_KERNEL_DIR/kernel.json"
{
  "argv": [
    "$JUPYTER_KERNEL_LAUNCHER",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$JUPYTER_KERNEL_DISPLAY",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
EOF

jupyter kernelspec list
```

Remove an obsolete kernel: `jupyter kernelspec uninstall -f <kernel_name>`. In VS Code: **Command Palette → Developer: Reload Window**.

---

## 9. Validate the Environment

```bash
source "$BASE_SCRATCH/Python4SmartSim.sh"

python -c "
import sys
import numpy, jax, equinox, tensorflow, torch, onnx, onnxruntime, pysr
import smartsim, smartredis

print(f'Python:       {sys.version.split()[0]}')
print(f'NumPy:        {numpy.__version__}')
print(f'JAX:          {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}')
print(f'Equinox:      {equinox.__version__}')
print(f'TensorFlow:   {tensorflow.__version__}')
print(f'PyTorch:      {torch.__version__}')
print(f'ONNX:         {onnx.__version__}')
print(f'ONNXRuntime:  {onnxruntime.__version__}')
print(f'PySR:         {pysr.__version__}')
print(f'SmartSim:     {smartsim.__version__}')
print(f'SmartRedis:   {smartredis.__version__}')
"
```

```bash
uv pip check
smart validate --device cpu
```

`smart validate` should report TensorFlow, ONNX Runtime, and LibTorch backends as available on both architectures.

**JuliaCall** (should not download or install anything — that already happened at build time):

```bash
python - <<'PY'
import juliapkg
from juliacall import Main as jl
print(f"Julia executable: {juliapkg.executable()}")
print(f"Julia version:    {jl.VERSION}")
PY
```

Native library check (both architectures):

```bash
ls -la "$SMARTREDIS_DIR/install/lib64"
test -f "$SMARTREDIS_DIR/install/lib64/libsmartredis-fortran.so" \
    && echo "SmartRedis Fortran library is available."
```

---

## 10. Dependency File Workflow

```text
requirements.in            Human-maintained direct dependencies (not SmartSim/SmartRedis themselves)
requirements-$ENV_ARCH.txt Installed-state snapshot (excludes SmartSim/SmartRedis)
```

**Add/remove a package** — edit `requirements.in`, then rebuild/update (Section 11 or 12). Removing a package needs a full rebuild to drop unused transitive deps.

**Keep `jax[cuda12]` unpinned** to pick up the newest compatible JAX release; `tensorflow` and `torch` remain pinned in `requirements.in`, while ONNX/ONNX Runtime/`protobuf`/`numpy` resolve at build time. Python-side TensorFlow/PyTorch/ONNX Runtime versions are independent of the RedisAI backend binaries `smart build` produces — exact equality isn't required, but validate exported models with `set_model`/`run_model` before production use.

**Reproduce an exact installed set** — temporarily point `extra4SmartSim.sh`'s two `uv pip install --requirements ...` lines at `requirements-$ENV_ARCH.txt`, rebuild, then switch back.

---

## 11. Updating the Environment

```bash
cat <<'EOF' > "$PYTHON_ROOT/update4SmartSim.sh"
#!/bin/bash
set -e

: "${CW_BUILD_TMPDIR:?CW_BUILD_TMPDIR is not set}"
: "${PYTHON_ROOT:?PYTHON_ROOT is not set}"
: "${ENV_ARCH:?ENV_ARCH is not set}"

export TMPDIR="$CW_BUILD_TMPDIR"
export PIP_CACHE_DIR="$CW_BUILD_TMPDIR/.pip_cache"
export UV_CACHE_DIR="$CW_BUILD_TMPDIR/.uv_cache"
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

python -m pip install --no-cache-dir uv

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"
if [ -s "$UPDATE_REQUEST" ]; then
    mapfile -t UPDATE_PACKAGES < "$UPDATE_REQUEST"
    uv pip install --link-mode=copy --upgrade "${UPDATE_PACKAGES[@]}"
fi

# Keep the packaged Julia environment ready for PySR
PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
export JULIA_DEPOT_PATH="$PYTHON_PREFIX/julia_depot"
export PYTHON_JULIAPKG_PROJECT="$PYTHON_PREFIX/julia_env"

python - <<'PY'
import juliapkg
import pysr
juliapkg.resolve()
print(f"PySR version:     {pysr.__version__}")
print(f"Julia executable: {juliapkg.executable()}")
PY

python - <<'PY'
import juliapkg, subprocess
julia, project = juliapkg.executable(), juliapkg.project()
subprocess.run(
    [julia, f"--project={project}", "-e",
     "using Pkg; Pkg.instantiate(); Pkg.precompile()"],
    check=True,
)
PY

uv pip install \
    --link-mode=copy \
    "smartredis @ git+https://github.com/PentagonToy/SmartRedis.git@csc-develop"

uv pip install \
    --link-mode=copy \
    "smartsim @ git+https://github.com/PentagonToy/SmartSim.git@csc-develop"

export USE_SYSTEMD=no

smart clobber

smart build \
    --device cpu \
    --skip-python-packages

uv pip install \
    --link-mode=copy \
    --requirements "$PYTHON_ROOT/requirements.in"

uv pip check

python -m pip list --format=freeze \
    | grep -v '^smartredis==' \
    | grep -v '^smartsim==' \
    | sort \
    > "$PYTHON_ROOT/requirements-$ENV_ARCH.txt"

rm -f "$UPDATE_REQUEST"
rm -rf "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
EOF

chmod +x "$PYTHON_ROOT/update4SmartSim.sh"
```

Create `smartsim-update`:

```bash
mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/smartsim-update"
#!/bin/bash -l
set -e

if [ "$#" -eq 0 ]; then
    echo "Usage: smartsim-update <package> [package ...]"
    exit 1
fi

if [ ! -f "$HOME/.config/csc-hpc/identity.sh" ]; then
    echo "Identity file not found: $HOME/.config/csc-hpc/identity.sh"
    exit 1
fi

source "$HOME/.config/csc-hpc/identity.sh"

export BASE_SCRATCH="/scratch/$CSC_PROJECT/$PROJECT_USER_DIR/Utilities"
export PYTHON_BASE="$BASE_SCRATCH/Python"
export PYTHON_ROOT="$PYTHON_BASE/PythonSmartSim"

case "$(uname -m)" in
    x86_64) export ENV_ARCH="x64" ;;
    aarch64) export ENV_ARCH="arm64" ;;
    *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

export ENV_PREFIX="$PYTHON_ROOT/envs/$ENV_NICKNAME-3.12-$ENV_ARCH"
export TMP_BUILD_DIR="$BASE_SCRATCH/.tykky_runtime_smartsim_$ENV_ARCH"
export UPDATE_REQUEST="$PYTHON_ROOT/.smartsim-update-$ENV_ARCH.txt"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Environment not found: $ENV_PREFIX"; exit 1
fi

if [ ! -f "$PYTHON_ROOT/requirements.in" ]; then
    echo "requirements.in not found: $PYTHON_ROOT/requirements.in"; exit 1
fi

for package in "$@"; do
    package_name="$(printf '%s\n' "$package" | sed -E 's/\[.*//; s/[<>=!~].*//')"
    case "$package_name" in
        smartsim|smartredis)
            echo "$package_name is managed separately and must not be added to requirements.in."
            exit 1
            ;;
    esac
done

printf '%s\n' "$@" > "$UPDATE_REQUEST"

python - "$PYTHON_ROOT/requirements.in" "$@" <<'PY'
import re, sys
from pathlib import Path

requirements_file = Path(sys.argv[1])
requested = sys.argv[2:]
lines = requirements_file.read_text().splitlines()

def package_name(spec):
    return re.split(r"[\[<>=!~]", spec, maxsplit=1)[0].strip().lower()

for spec in requested:
    name = package_name(spec)
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or " @ " in stripped:
            continue
        if package_name(stripped) == name:
            lines[index] = spec
            replaced = True
            print(f"Updated requirement: {spec}")
            break
    if not replaced:
        lines.append(spec)
        print(f"Added requirement: {spec}")

requirements_file.write_text("\n".join(lines) + "\n")
PY

module purge
module load tykky

export TMPDIR="$TMP_BUILD_DIR"
export CW_BUILD_TMPDIR="$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"

conda-containerize update \
    --post-install "$PYTHON_ROOT/update4SmartSim.sh" \
    "$ENV_PREFIX"

echo "Update completed. Recorded packages: $PYTHON_ROOT/requirements-$ENV_ARCH.txt"
EOF

chmod +x "$HOME/bin/smartsim-update"
```

```bash
grep -qxF 'export PATH="$HOME/bin:$PATH"' ~/.bashrc || \
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Apply on the matching architecture node:

```bash
smartsim-update pydantic
smartsim-update loguru pyinstrument
```

Updating does **not** rebuild the native SmartRedis library, and does **not** refresh the writable Julia runtime copy under `.julia_env_runtime_$ENV_ARCH` — since `conda-containerize update` repacks the Julia project inside the container, re-run the copy step in Section 5.1 afterwards if PySR's Julia dependencies changed.

---

## 12. Rebuild / Clean Reinstall

```bash
# 1) Run the matching Global Configuration block (Section 1) first.
echo "ENV_ARCH=$ENV_ARCH"; echo "ENV_PREFIX=$ENV_PREFIX"; echo "SMARTREDIS_DIR=$SMARTREDIS_DIR"

rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$PYTHON_ROOT/envs" "$TMP_BUILD_DIR"
# Also clear the writable Julia runtime copy, since it's derived from this build:
rm -rf "$BASE_SCRATCH/.julia_env_runtime_$ENV_ARCH" "$BASE_SCRATCH/.julia_depot_runtime_$ENV_ARCH"
# For a full clean install, also: rm -rf "$SMARTREDIS_DIR"

ls -l "$PYTHON_ROOT/base4SmartSim.yml" "$PYTHON_ROOT/requirements.in" "$PYTHON_ROOT/extra4SmartSim.sh"
chmod +x "$PYTHON_ROOT/extra4SmartSim.sh"
```

Request a node (Section 4), then build (Section 5), followed immediately by the one-time Julia runtime preparation (Section 5.1). If you removed `$SMARTREDIS_DIR`, rebuild it too (Section 6), which also rewrites `runtime-$ENV_ARCH.sh`.

---

## 13. Troubleshooting

**Total reset:**
```bash
rm -rf "$ENV_PREFIX" "$TMP_BUILD_DIR"
mkdir -p "$TMP_BUILD_DIR"
```
Rebuild per Section 12.

**`requirements-$ENV_ARCH.txt` missing** — only written after a successful build; run Section 5.

**PySR tries to download Julia/packages at runtime instead of using the precompiled build** — confirm `$JULIA_ENV_RUNTIME` (Section 5.1) exists and that `PYTHON_JULIAPKG_OFFLINE=yes` is set: `python -c "import os; print(os.environ.get('PYTHON_JULIAPKG_OFFLINE'))"` should print `yes`.

**`OSError: Read-only file system` when importing `pysr`** — the one-time Julia-project copy (Section 5.1) was never run for this architecture, or its target directory was deleted. Re-run Section 5.1, then re-source `Python4SmartSim.sh`; the loader itself no longer performs any copy and will refuse to load if `$JULIA_ENV_RUNTIME` is missing.

**"This file must be sourced, not executed" when running the loader** — run it with `source Python4SmartSim.sh`, not `./Python4SmartSim.sh` or `bash Python4SmartSim.sh`.

**`PATH`/`LD_LIBRARY_PATH` grows every time the loader is sourced** — this should no longer happen; the loader's `path_prepend` helper checks for existing entries before prepending. If it does happen, confirm you're using the updated loader from Section 7, not an older version.

**`git+ ...@csc-develop` install fails** — confirm outbound network access from the build node; confirm the branch name is spelled correctly.

**`smart build` reports incompatible-pointer-type compile errors** — retry with `CFLAGS="-Wno-incompatible-pointer-types" CXXFLAGS="-Wno-incompatible-pointer-types"` prefixed to `smart clobber`/`smart build`.

**`smart build` rejects `--skip-python-packages`** — run `smart build --help` inside the build environment to get the ground-truth flags for whatever version is installed.

**TensorFlow/PyTorch/ONNX Runtime model compatibility with RedisAI** — Python package versions and RedisAI backend versions are separate and don't need to match exactly. Validate actual exported TensorFlow, TorchScript, and ONNX models with `set_model`/`run_model`. If a model uses operators/formats the RedisAI backend doesn't support, re-export with an older compatible opset or framework format.

**uv hardlink warning** — expected; `--link-mode=copy` handles it.

**Home quota exceeded during build** — caches redirect to `$BASE_SCRATCH/.tykky_runtime_smartsim_*`, not `$HOME`.

**Architecture mismatch** — build and use the matching-architecture Tykky environment; no cross-architecture container.

**JAX reports no GPU** — loader sets `JAX_PLATFORMS` automatically; avoid `JAX_PLATFORMS=gpu`.

**SmartRedis native library not found** — check `$LD_LIBRARY_PATH`, confirm files under `install/lib64` (or `lib`), re-source the loader.

**Jupyter kernel doesn't see the same environment as the terminal** — confirm `kernel.json`'s `argv` points at `jupyter-kernel-$ENV_ARCH.sh` (Section 8), not directly at `$ENV_PREFIX/bin/python`; the wrapper sources the loader before launching `ipykernel_launcher`.

**Import errors after an update** — run `uv pip check`; prefer a full rebuild (Section 12) over stacking updates.

**Identity file not found** — go back to Section 0.

---

## 14. SmartSim Deployment Track

Each architecture runs its own local Orchestrator, and can also run JAX/Equinox/TensorFlow/PyTorch/PySR workloads locally:

```text
x64 CPU node                              arm64 GPU node
└─ SmartSim Orchestrator                  └─ SmartSim Orchestrator
   (Redis + RedisAI: TF/ONNX/LibTorch)       (Redis + RedisAI: TF/ONNX/LibTorch)
   └─ tensor/weight/metric storage            └─ JAX/Equinox/TF/PyTorch/PySR training & inference
                                                  + tensor/weight/metric storage
```

RedisAI model execution is available but optional. TensorFlow, ONNX, and PyTorch (via LibTorch) models can be executed directly inside RedisAI with `set_model`/`run_model`; the primary workflow may still run JAX/Equinox/PySR inference in external Python workers, with SmartRedis carrying tensors either way:

```python
from smartredis import Client
import jax.numpy as jnp

client = Client(address="localhost:6379", cluster=False)
x = jnp.asarray(client.get_tensor("training_data"))
result = jax_function(x)          # runs on the GPU
client.put_tensor("result", result)
```

Other typical workflows: launching OpenFOAM solvers + Python producers/consumers through Slurm; linking external C++/Fortran solvers against the native SmartRedis client; symbolic regression on simulation output with PySR; validating producer/consumer config with `pydantic`, logging with `loguru`, profiling with `pyinstrument`; DVC-tracked datasets; Papermill-driven notebook pipelines.

Full production architecture and Slurm templates: [SmartSim4CSC](https://github.com/PentagonToy/SmartSim4CSC).

---

## Notes

* Python 3.12, built separately per architecture — never mix containers across architectures.
* **This environment is now a superset of the previously standalone ML stack.** It includes everything the ML environment had — including PySR/Julia — plus SmartSim/SmartRedis and RedisAI's TensorFlow/ONNX Runtime/LibTorch backends. A separate `PythonML/` environment is not needed if you use this stack.
* SmartSim and SmartRedis install from the CSC forks (`PentagonToy/SmartSim`, `PentagonToy/SmartRedis`, `csc-develop` branch), not PyPI — no runtime patching remains in `extra4SmartSim.sh` / `update4SmartSim.sh`.
* **PySR's Julia dependency is resolved and precompiled at build time**, exactly as in the standalone ML stack, and the writable runtime copy of the Julia project is now created **once**, immediately after a successful Tykky build (Section 5.1) — not on every `source`. The Julia depot (precompiled packages) stays read-only and is layered in via `JULIA_DEPOT_PATH`. `PYTHON_JULIAPKG_OFFLINE=yes` prevents any runtime re-download.
* The loader (`Python4SmartSim.sh`) is a **pure loader**: it must be sourced (not executed), is idempotent across repeated sourcing thanks to a `path_prepend` helper, and only loads the GCC module recorded in `runtime-$ENV_ARCH.sh` if it isn't already loaded.
* Jupyter kernels run through a **launcher wrapper** (`jupyter-kernel-$ENV_ARCH.sh`) that sources the loader before starting `ipykernel_launcher`, so notebook kernels — including under VS Code — always match an interactive terminal session rather than duplicating environment variables inside `kernel.json`.
* `smart build` runs identically on both architectures and builds all three RedisAI backends by default; `--skip-python-packages` is used because TensorFlow/PyTorch/ONNX Python packages are already managed via `requirements.in`.
* `jax[cuda12]`, `onnx`, `onnxruntime`, `numpy`, and `protobuf` are intentionally unpinned and resolve to the newest compatible versions at build time. The exact installed versions are recorded in `requirements-$ENV_ARCH.txt` — validate the resolved environment with `uv pip check`.
* Placeholders (`Harry`/`Dumbledore`/`project_xxxxxxx`) are set once in the identity file (Section 0).
* `requirements.in` = direct deps, not SmartSim/SmartRedis themselves; `requirements-$ENV_ARCH.txt` = installed-state snapshot, not a lockfile.
* Every `uv pip install` uses `--link-mode=copy`.
* The native SmartRedis library (Section 6) is unrelated to `smart build`/RedisAI and the Julia toolchain, and is built on both architectures as usual. The GCC module needed for it varies by target system (Roihu, Mahti, Puhti) and is now recorded once in `runtime-$ENV_ARCH.sh` at build time, rather than guessed from hostname at every `source`.
