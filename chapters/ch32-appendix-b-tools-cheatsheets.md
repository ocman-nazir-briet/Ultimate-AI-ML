[← Appendix A: Mathematical Notation and Reference](ch31-appendix-a-math-notation.md) · [Contents](../README.md#contents) · [Appendix C: Datasets, Projects, and Further Reading →](ch33-appendix-c-datasets-resources.md)

---

# Appendix B: Tools, Environment Setup, and Cheat Sheets

This appendix is a desk reference. Where the main chapters explain *why* a tool works the way it does, the pages that follow are meant to answer *how* — quickly, while your hands are on the keyboard. Keep it open in a second window. The material is organized so you can jump straight to a task: set up an environment, look up a NumPy broadcasting rule, remember the argument order of a pandas merge, recall the estimator API, or paste a minimal training loop and adapt it.

Nothing here is exhaustive — the official documentation for each library is the authoritative source, and it changes faster than any printed page. What this appendix gives you is the *working subset*: the calls, patterns, and idioms that come up daily, arranged so the common case is one glance away. The chapters that introduce these tools in depth — **Python Tooling for Machine Learning**, **The Machine Learning Workflow**, **Neural Network Fundamentals**, **How Large Language Models Work**, and **LLMs in Practice** — remain the place to build understanding. This is the place to refresh it.

A note on running the examples: the fenced code blocks are illustrative and are not executed by the build. They are written to be correct and copy-pasteable, but versions drift; if a call signature has changed since printing, trust the library’s own help (`help(obj)`, `obj?` in IPython, or the online docs) over the page.

## 30.25 Environment Setup

Every reproducible ML project starts with an isolated, declared environment. The single most common source of “works on my machine” failures is an undeclared dependency or a version mismatch, and every tool below exists to eliminate that class of problem.

### 30.25.1 Installing Python

Do not use the system Python that ships with your operating system for project work — it is managed by the OS and modifying it can break system tools. Install a separate interpreter you control.

| Approach | Command / source | When to use |
|----|----|----|
| **python.org installer** | Download from python.org | Simplest single-version install on Windows/macOS |
| **pyenv** | `curl https://pyenv.run \| bash` | Manage many Python versions side by side |
| **uv** (installs Pythons too) | `uv python install 3.12` | Modern all-in-one; fetches interpreters on demand |
| **conda / miniconda** | miniconda installer | Scientific stack with non-Python deps (CUDA, MKL) |
| **Homebrew (macOS)** | `brew install python@3.12` | Quick, but pin versions carefully |

``` bash
# Check what you have and where it lives
python3 --version        # -> Python 3.12.4
which python3            # -> /usr/bin/python3  (avoid using this one for projects)

# With pyenv: install and pin a version for the current directory
pyenv install 3.12.4
pyenv local 3.12.4       # writes .python-version
python --version         # -> Python 3.12.4
```

**Rule of thumb:** target a Python version one or two minor releases behind the very latest. The newest release often lacks prebuilt wheels for heavy libraries (NumPy, PyTorch, TensorFlow), forcing slow source builds or outright failures.

### 30.25.2 Virtual environments

A **virtual environment** is a self-contained directory holding a specific interpreter and its own installed packages, isolated from every other project. Activate it, and `python` and `pip` resolve to that sandbox.

**venv** — built into the standard library, zero extra install:

``` bash
python -m venv .venv                 # create env in ./.venv
source .venv/bin/activate            # activate (macOS/Linux)
# .venv\Scripts\activate            # activate (Windows PowerShell)
pip install numpy pandas scikit-learn
deactivate                           # leave the environment
```

**conda** — manages Python *and* non-Python binaries (compilers, CUDA runtimes, MKL). Indispensable when a dependency has heavy native components:

``` bash
conda create -n myproj python=3.12   # named environment
conda activate myproj
conda install numpy pandas scikit-learn
conda env export > environment.yml   # capture full spec
conda deactivate
```

**uv** — a fast, modern package and environment manager that resolves and installs in seconds and can manage interpreters, virtual environments, and lockfiles together:

``` bash
uv venv                              # create .venv with the project's Python
uv pip install numpy pandas          # install into it (no manual activate needed)
uv add scikit-learn                  # add to pyproject.toml AND install AND lock
uv run python train.py               # run inside the managed env
```

The three tools compared:

| Tool | Speed | Manages interpreters | Non-Python deps | Lockfile | Best for |
|----|----|----|----|----|----|
| **venv + pip** | Baseline | No | No | via `pip freeze` | Simplicity, no extra install |
| **conda** | Slow solver | Yes | **Yes** (CUDA, MKL) | `environment.yml` | GPU stacks, scientific binaries |
| **uv** | **Very fast** | Yes | No | `uv.lock` (built in) | Modern pure-Python workflows |

### 30.25.3 Managing dependencies and lockfiles

There is a crucial distinction between the packages you *ask for* and the exact versions you *get*. Declaring `pandas` gets you whatever the resolver picks today; six months later a teammate gets a different version. A **lockfile** records the exact resolved version (and hash) of every package, direct and transitive, so the environment can be rebuilt bit-for-bit.

| File | Role | Produced by |
|----|----|----|
| `requirements.txt` | Loose or pinned direct deps | Hand-written or `pip freeze` |
| `pyproject.toml` | Declared project deps + metadata | Hand-written / `uv add` |
| `uv.lock` | Fully resolved, hashed lock | `uv lock` |
| `environment.yml` | conda spec (can be loose or pinned) | `conda env export` |
| `poetry.lock` | Poetry’s resolved lock | `poetry lock` |

``` bash
# pip: capture the exact current state (a crude lock)
pip freeze > requirements.txt
pip install -r requirements.txt      # reproduce elsewhere

# pyproject.toml declaring direct dependencies
# [project]
# dependencies = ["numpy>=1.26", "pandas>=2.1", "scikit-learn>=1.4"]

# uv: declared deps in pyproject.toml, exact versions in uv.lock
uv lock                              # resolve + write uv.lock
uv sync                              # make the env match the lock exactly
```

**The reproducibility rule:** commit *both* the loose declaration (what you want) and the lockfile (what you got). Rebuild from the lock in CI and on teammates’ machines; regenerate the lock deliberately when you intend to upgrade.

### 30.25.4 Reproducible environments — a checklist

Environment reproducibility is necessary but not sufficient for *result* reproducibility. A complete recipe pins more than packages:

- **Pin the Python version** — record it in `.python-version` or `pyproject.toml`.
- **Commit a lockfile** — exact versions of every transitive dependency.
- **Seed every source of randomness** — see the snippet below.
- **Record hardware/driver versions** for GPU work — CUDA and cuDNN versions affect numerics and reproducibility.
- **Containerize for hard guarantees** — a Docker image freezes the OS, system libraries, and the whole stack.
- **Version data and models**, not just code — see the git-for-ML section.

``` python
import os, random
import numpy as np

def set_seeds(seed: int = 42) -> None:
    """Seed the common RNGs used across the stack."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # tf.random.set_seed(seed)

set_seeds(42)
# Note: exact GPU reproducibility also needs deterministic kernels,
# e.g. torch.use_deterministic_algorithms(True), which can slow training.
```

A minimal Dockerfile that locks the whole stack:

``` dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train.py"]
```

### 30.25.5 GPU and CUDA setup notes

GPUs accelerate the dense linear algebra at the heart of deep learning, but the software stack is fussy. The layers must agree:

| Layer | What it is | How to check |
|----|----|----|
| **NVIDIA driver** | Kernel-level GPU driver | `nvidia-smi` |
| **CUDA toolkit** | GPU compute runtime/libraries | `nvcc --version` |
| **cuDNN** | Deep-learning primitives on CUDA | reported by framework |
| **Framework build** | PyTorch/TF compiled for a CUDA version | `torch.version.cuda` |

The most reliable path is to **install the framework build that bundles its own CUDA runtime**, so you only need a recent enough driver — not a hand-matched system CUDA toolkit.

``` python
# nvidia-smi shows driver version and the max CUDA it supports (top-right)
nvidia-smi

# PyTorch: pick the build for your CUDA line from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu124
import torch
print(torch.cuda.is_available())     # True if a usable GPU is visible
print(torch.cuda.device_count())     # number of GPUs
print(torch.cuda.get_device_name(0)) # e.g. "NVIDIA A100-SXM4-40GB"
print(torch.version.cuda)            # CUDA version this build targets
```

On **Apple Silicon**, PyTorch uses the Metal backend: the device string is `"mps"` rather than `"cuda"`. On machines with no GPU, everything still runs on `"cpu"` — slower, but correct.

### 30.25.6 Jupyter and Colab

**Jupyter** notebooks interleave code, output, and prose — ideal for exploration, teaching, and reporting. Install into your project environment so the kernel sees your packages:

``` bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name myproj --display-name "Python (myproj)"
jupyter lab                          # launch in the browser
```

Handy notebook idioms (IPython “magics”):

| Magic | Effect |
|----|----|
| `%timeit expr` | Time a single expression over many runs |
| `%%time` | Time a whole cell (top of cell) |
| `%matplotlib inline` | Render plots in the notebook |
| `%load_ext autoreload` / `%autoreload 2` | Auto-reimport edited modules |
| `!pip install pkg` | Run a shell command from a cell |
| `obj?` / `obj??` | Show docstring / source |

**Google Colab** is a hosted notebook with free (and paid) GPUs — the fastest way to get onto an accelerator with zero local setup. Select the runtime type (**Runtime → Change runtime type → GPU**), then:

``` python
# In a Colab cell — install extras and mount Drive for persistence
!pip install -q transformers
import torch; print(torch.cuda.get_device_name(0))
from google.colab import drive; drive.mount("/content/drive")
```

Colab’s caveats: sessions are ephemeral (files vanish when the runtime recycles — save to Drive), GPU availability varies on the free tier, and long jobs can be interrupted. It is excellent for experiments and prototyping, less so for multi-day training runs.

## 30.26 NumPy Cheat Sheet

**NumPy** provides the `ndarray`, the n-dimensional array that is the common currency of the entire Python numerical stack. Its power comes from **vectorization**: expressing computation as whole-array operations that dispatch to fast compiled kernels instead of Python loops. See **Python Tooling for Machine Learning** for the conceptual treatment.

``` python
import numpy as np
```

### 30.26.1 Array creation

| Call | Result |
|----|----|
| `np.array([1, 2, 3])` | Array from a Python list |
| `np.zeros((2, 3))` | 2×3 of zeros |
| `np.ones((2, 3))` | 2×3 of ones |
| `np.full((2, 2), 7)` | 2×2 filled with 7 |
| `np.eye(3)` | 3×3 identity matrix |
| `np.arange(0, 10, 2)` | `[0 2 4 6 8]` (start, stop, step) |
| `np.linspace(0, 1, 5)` | 5 evenly spaced points in `[0, 1]` |
| `np.random.default_rng(42).random((2, 2))` | 2×2 of uniform random floats |
| `np.random.default_rng(42).normal(0, 1, 100)` | 100 draws from N(0, 1) |

``` python
rng = np.random.default_rng(42)      # the modern, seedable RNG
a = np.arange(12)                    # array([0, 1, ..., 11])
b = np.linspace(0.0, 1.0, 5)         # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

### 30.26.2 Shape, dtype, and reshaping

| Attribute / method          | Meaning                                 |
|-----------------------------|-----------------------------------------|
| `a.shape`                   | Tuple of dimension sizes, e.g. `(3, 4)` |
| `a.ndim`                    | Number of dimensions                    |
| `a.size`                    | Total number of elements                |
| `a.dtype`                   | Element type, e.g. `float64`, `int32`   |
| `a.reshape(3, 4)`           | New view with a different shape         |
| `a.reshape(-1, 4)`          | `-1` infers that dimension              |
| `a.ravel()` / `a.flatten()` | Flatten to 1-D (view / copy)            |
| `a.T` / `a.transpose()`     | Transpose axes                          |
| `a[:, np.newaxis]`          | Insert a new axis (for broadcasting)    |
| `a.astype(np.float32)`      | Cast to another dtype                   |

``` python
a = np.arange(12)
grid = a.reshape(3, 4)               # shape (3, 4)
col  = a.reshape(-1, 1)              # shape (12, 1), a column vector
print(grid.shape, grid.dtype)        # (3, 4) int64
```

### 30.26.3 Indexing and slicing

NumPy slicing returns **views** (no copy) where possible; fancy and boolean indexing return copies.

| Expression            | Selects                                         |
|-----------------------|-------------------------------------------------|
| `a[2]`                | Single element (1-D)                            |
| `a[1:4]`              | Slice, elements 1–3                             |
| `a[::2]`              | Every other element                             |
| `M[1, 2]`             | Row 1, column 2                                 |
| `M[1, :]` / `M[:, 2]` | Whole row 1 / whole column 2                    |
| `M[0:2, 1:3]`         | Sub-block (rows 0–1, cols 1–2)                  |
| `a[a > 0]`            | **Boolean mask** — elements passing a condition |
| `a[[0, 2, 4]]`        | **Fancy indexing** — pick specific indices      |
| `M[M > 0] = 0`        | Assign through a mask                           |

``` python
M = np.arange(12).reshape(3, 4)
print(M[1, :])          # [4 5 6 7]  -> row 1
print(M[:, 2])          # [ 2 6 10]  -> column 2
mask = M % 2 == 0
print(M[mask])          # [ 0  2  4  6  8 10]  -> even entries
```

### 30.26.4 Broadcasting

**Broadcasting** lets NumPy operate on arrays of different shapes by virtually stretching the smaller one — without copying data. The rule: align shapes from the **right**; dimensions are compatible when they are equal or one of them is 1.

``` python
A = np.ones((3, 4))          # shape (3, 4)
row = np.arange(4)           # shape (4,)   -> broadcasts across rows
col = np.arange(3).reshape(3, 1)  # shape (3, 1) -> broadcasts across cols

print((A + row).shape)       # (3, 4)   row added to every row
print((A + col).shape)       # (3, 4)   col added to every column

# Standardize columns of a data matrix: (X - mean) / std, broadcast over rows
X = np.random.default_rng(0).normal(size=(100, 5))
X_std = (X - X.mean(axis=0)) / X.std(axis=0)   # means/stds shape (5,) broadcast
```

Shape-compatibility examples:

| Shape A  | Shape B  | Result    | Note                          |
|----------|----------|-----------|-------------------------------|
| `(3, 4)` | `(4,)`   | `(3, 4)`  | B stretched over rows         |
| `(3, 4)` | `(3, 1)` | `(3, 4)`  | B stretched over columns      |
| `(3, 1)` | `(1, 4)` | `(3, 4)`  | Both stretched (outer-like)   |
| `(3, 4)` | `(3,)`   | **error** | Right-aligned 4 vs 3 mismatch |

### 30.26.5 Common operations and reductions

| Operation          | Call                                                   |
|--------------------|--------------------------------------------------------|
| Elementwise math   | `a + b`, `a * b`, `np.sqrt(a)`, `np.exp(a)`            |
| Sum / mean / std   | `a.sum()`, `a.mean()`, `a.std()`                       |
| Along an axis      | `M.sum(axis=0)` (down columns), `axis=1` (across rows) |
| Min / max / argmax | `a.min()`, `a.max()`, `a.argmax()`                     |
| Cumulative         | `a.cumsum()`, `a.cumprod()`                            |
| Sorting            | `np.sort(a)`, `a.argsort()`                            |
| Unique values      | `np.unique(a, return_counts=True)`                     |
| Clip to range      | `np.clip(a, 0, 1)`                                     |
| Stack arrays       | `np.vstack`, `np.hstack`, `np.concatenate`             |
| Conditional select | `np.where(cond, x, y)`                                 |

``` python
M = np.arange(6).reshape(2, 3)
print(M.sum())            # 15   -> everything
print(M.sum(axis=0))      # [3 5 7]  -> one value per column
print(M.sum(axis=1))      # [3 12]   -> one value per row
```

**Axis intuition:** `axis=0` collapses the *rows* (giving a per-column result); `axis=1` collapses the *columns* (a per-row result). The collapsed axis disappears from the output shape.

### 30.26.6 Linear algebra

| Call                         | Computes                           |
|------------------------------|------------------------------------|
| `A @ B` or `np.matmul(A, B)` | Matrix product                     |
| `np.dot(a, b)`               | Dot product (1-D) / matrix product |
| `np.linalg.inv(A)`           | Matrix inverse                     |
| `np.linalg.solve(A, b)`      | Solve `Ax = b` (prefer over `inv`) |
| `np.linalg.det(A)`           | Determinant                        |
| `np.linalg.eig(A)`           | Eigenvalues and eigenvectors       |
| `np.linalg.svd(A)`           | Singular value decomposition       |
| `np.linalg.norm(a)`          | Vector/matrix norm                 |
| `A.T`                        | Transpose                          |

``` python
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)            # solve Ax = b -> array([2., 3.])
# Verify: A @ x should equal b
assert np.allclose(A @ x, b)
```

Prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b` — it is both faster and numerically more stable. The mathematics behind these operations lives in **Mathematical Foundations**.

## 30.27 pandas Cheat Sheet

**pandas** provides the `DataFrame`, a labeled, heterogeneous table built on NumPy. It is the workhorse for loading, cleaning, and reshaping tabular data before it reaches a model.

``` python
import pandas as pd
```

### 30.27.1 Input and output

| Read                           | Write                             |
|--------------------------------|-----------------------------------|
| `pd.read_csv("f.csv")`         | `df.to_csv("f.csv", index=False)` |
| `pd.read_parquet("f.parquet")` | `df.to_parquet("f.parquet")`      |
| `pd.read_excel("f.xlsx")`      | `df.to_excel("f.xlsx")`           |
| `pd.read_json("f.json")`       | `df.to_json("f.json")`            |
| `pd.read_sql(query, conn)`     | `df.to_sql("table", conn)`        |

``` python
df = pd.read_csv("sales.csv", parse_dates=["date"])
df.to_parquet("sales.parquet")       # Parquet is columnar, compact, typed
```

**Prefer Parquet** for intermediate data: it is columnar, compressed, and preserves dtypes, so round-trips are lossless and fast. CSV is universal but slow, untyped, and lossy.

### 30.27.2 Inspecting a DataFrame

| Call                         | Shows                                 |
|------------------------------|---------------------------------------|
| `df.head(n)` / `df.tail(n)`  | First / last `n` rows                 |
| `df.shape`                   | `(rows, columns)`                     |
| `df.info()`                  | Dtypes, non-null counts, memory       |
| `df.describe()`              | Summary statistics of numeric columns |
| `df.dtypes`                  | Column types                          |
| `df.columns`                 | Column labels                         |
| `df["col"].value_counts()`   | Frequency of each value               |
| `df.isna().sum()`            | Missing count per column              |
| `df.memory_usage(deep=True)` | Memory per column                     |

### 30.27.3 Selection and filtering

The central distinction: `.loc` indexes by *label*, `.iloc` by *integer position*.

| Expression                              | Selects                         |
|-----------------------------------------|---------------------------------|
| `df["col"]`                             | One column (a Series)           |
| `df[["a", "b"]]`                        | Multiple columns (a DataFrame)  |
| `df.loc[3]`                             | Row with label 3                |
| `df.loc[df["age"] > 30]`                | Rows passing a condition        |
| `df.loc[df["age"] > 30, "name"]`        | Filtered rows, one column       |
| `df.iloc[0:5]`                          | First five rows by position     |
| `df.iloc[0, 2]`                         | Row 0, column 2 by position     |
| `df.query("age > 30 and city == 'NY'")` | Filter with a string expression |
| `df.sample(n=5, random_state=0)`        | Random sample of rows           |

``` python
adults = df.loc[df["age"] >= 18]                   # boolean row filter
subset = df.loc[df["age"] >= 18, ["name", "age"]]  # rows + specific columns
by_pos = df.iloc[:10, :3]                          # first 10 rows, first 3 cols
```

### 30.27.4 Common column operations

| Task                | Call                                       |
|---------------------|--------------------------------------------|
| New column          | `df["c"] = df["a"] + df["b"]`              |
| Rename              | `df.rename(columns={"old": "new"})`        |
| Drop                | `df.drop(columns=["a", "b"])`              |
| Change dtype        | `df["a"] = df["a"].astype("float32")`      |
| Sort by column      | `df.sort_values("age", ascending=False)`   |
| Map/replace values  | `df["c"].map({0: "no", 1: "yes"})`         |
| Bin into categories | `pd.cut(df["age"], bins=[0, 18, 65, 120])` |
| One-hot encode      | `pd.get_dummies(df, columns=["city"])`     |

### 30.27.5 Missing data

| Call                               | Effect                            |
|------------------------------------|-----------------------------------|
| `df.isna()` / `df.notna()`         | Boolean mask of missing / present |
| `df.dropna()`                      | Drop rows with any NaN            |
| `df.dropna(subset=["a"])`          | Drop rows where `a` is NaN        |
| `df.fillna(0)`                     | Fill NaN with a constant          |
| `df["a"].fillna(df["a"].median())` | Fill with the column median       |
| `df.ffill()` / `df.bfill()`        | Forward / backward fill           |
| `df.interpolate()`                 | Fill by interpolation             |

``` python
df["income"] = df["income"].fillna(df["income"].median())  # robust imputation
df = df.dropna(subset=["target"])                          # never impute the label
```

**Guideline:** impute *features* thoughtfully (median for skewed numerics, a sentinel or mode for categoricals), but never fabricate the *target*. Do imputation inside a pipeline so the fill values are learned on the training fold only — see the scikit-learn section and **The Machine Learning Workflow** for why this matters for leakage.

### 30.27.6 GroupBy — split, apply, combine

`groupby` splits rows into groups, applies an aggregation to each, and combines the results.

``` python
# One aggregation per group
df.groupby("city")["income"].mean()

# Multiple aggregations, multiple columns
df.groupby("city").agg(
    avg_income=("income", "mean"),
    max_age=("age", "max"),
    n=("age", "size"),
)

# Group by several keys
df.groupby(["city", "gender"])["income"].median()

# Transform: broadcast a group statistic back to every row (for features)
df["income_vs_city"] = df["income"] - df.groupby("city")["income"].transform("mean")
```

| Reducer | Meaning |
|----|----|
| `.mean()`, `.median()`, `.sum()` | Central tendency / total |
| `.count()` / `.size()` | Non-null count / group size |
| `.min()`, `.max()`, `.std()` | Extremes / spread |
| `.agg([...])` | Several reducers at once |
| `.transform(fn)` | Return group stat aligned to original rows |
| `.apply(fn)` | Arbitrary per-group function (slowest) |

### 30.27.7 Merge, join, and concatenate

`pd.merge` is the SQL-style join; `pd.concat` stacks frames.

``` python
merged = pd.merge(orders, customers, on="customer_id", how="left")

# Different key names on each side
pd.merge(a, b, left_on="uid", right_on="user_id", how="inner")

# Stack rows (same columns) / columns (same index)
pd.concat([jan, feb, mar], axis=0, ignore_index=True)   # more rows
pd.concat([features, extra_features], axis=1)            # more columns
```

| `how=`    | Keeps                                       |
|-----------|---------------------------------------------|
| `"inner"` | Only keys present in **both** frames        |
| `"left"`  | All left rows; matching right (NaN if none) |
| `"right"` | All right rows; matching left               |
| `"outer"` | Union of keys from both                     |

A frequent gotcha: after a `"left"` merge, unmatched right-side columns are NaN — check `merged.isna().sum()` to confirm the match rate is what you expected.

### 30.27.8 apply, map, and vectorization

| Call | Scope |
|----|----|
| `s.map(fn)` | Elementwise over a Series |
| `df.apply(fn, axis=1)` | Row-wise (slow — avoid on big data) |
| `df.apply(fn, axis=0)` | Column-wise |
| `s.str.lower()`, `s.str.contains("x")` | Vectorized string ops |
| `s.dt.year`, `s.dt.dayofweek` | Vectorized datetime accessors |

``` python
# Prefer vectorized ops over apply where possible — they are far faster:
df["name"] = df["name"].str.strip().str.title()     # vectorized string cleanup
df["year"] = df["date"].dt.year                       # vectorized datetime
# Reach for .apply only when no vectorized equivalent exists.
```

## 30.28 Matplotlib Cheat Sheet

**Matplotlib** is the foundational plotting library. Its object model has two levels: a **Figure** (the whole canvas) contains one or more **Axes** (individual plots). Prefer the explicit object-oriented interface (`fig, ax = plt.subplots()`) over the implicit `plt.plot()` state machine — it scales cleanly to multi-panel figures.

``` python
import matplotlib.pyplot as plt
```

### 30.28.1 Figure and axes

``` python
fig, ax = plt.subplots(figsize=(6, 4))       # one plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid; axes is a 2-D array

ax.plot(x, y)
ax.set_title("Title")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend()
fig.tight_layout()          # prevent label overlap
fig.savefig("plot.png", dpi=150, bbox_inches="tight")
```

### 30.28.2 Common plot types

| Call                             | Plot                          |
|----------------------------------|-------------------------------|
| `ax.plot(x, y)`                  | Line plot                     |
| `ax.scatter(x, y)`               | Scatter plot                  |
| `ax.bar(x, h)` / `ax.barh(y, w)` | Vertical / horizontal bars    |
| `ax.hist(data, bins=30)`         | Histogram                     |
| `ax.boxplot(data)`               | Box-and-whisker               |
| `ax.imshow(matrix)`              | Heatmap / image               |
| `ax.contourf(X, Y, Z)`           | Filled contour                |
| `ax.errorbar(x, y, yerr=e)`      | Points with error bars        |
| `ax.fill_between(x, lo, hi)`     | Shaded band (e.g. confidence) |

``` python
import numpy as np
x = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos", linestyle="--")
ax.set_title("Trig functions"); ax.legend()
fig.savefig("trig.png", dpi=150)
```

### 30.28.3 Styling

| Task | Call |
|----|----|
| Axis limits | `ax.set_xlim(0, 10)`, `ax.set_ylim(-1, 1)` |
| Log scale | `ax.set_yscale("log")` |
| Grid | `ax.grid(True, alpha=0.3)` |
| Ticks | `ax.set_xticks([...])`, `ax.set_xticklabels([...])` |
| Text annotation | `ax.annotate("peak", xy=(x, y), xytext=(x2, y2))` |
| Line style / color | `ax.plot(x, y, color="C1", linewidth=2, linestyle=":")` |
| Global style | `plt.style.use("seaborn-v0_8-darkgrid")` |
| Shared axes | `plt.subplots(2, 1, sharex=True)` |

**Seaborn** builds statistical, DataFrame-aware plots on top of Matplotlib in one line: `sns.histplot(df, x="age", hue="group")`, `sns.heatmap(corr, annot=True)`, `sns.pairplot(df)`. It returns Matplotlib Axes, so all styling above still applies.

## 30.29 scikit-learn Cheat Sheet

**scikit-learn** unifies classical machine learning behind one consistent **estimator API**. Learning that API once means you can swap models with almost no code change — the conceptual payoff explored in **The Machine Learning Workflow** and the supervised chapters.

``` python
from sklearn.model_selection import train_test_split
```

### 30.29.1 The estimator API

Every estimator follows the same protocol:

| Method | Purpose | Available on |
|----|----|----|
| `est.fit(X, y)` | Learn parameters from data | All estimators |
| `est.predict(X)` | Predict labels/values | Classifiers, regressors |
| `est.predict_proba(X)` | Class probabilities | Most classifiers |
| `est.transform(X)` | Transform features | Transformers (scalers, PCA) |
| `est.fit_transform(X)` | Fit then transform in one call | Transformers |
| `est.score(X, y)` | Default metric (accuracy / R²) | Predictors |

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = SomeEstimator()
model.fit(X_train, y_train)          # learn
preds = model.predict(X_test)        # predict
acc = model.score(X_test, y_test)    # evaluate
```

The golden rule: call `fit` (or `fit_transform`) on **training data only**; call `transform`/`predict` on test data. Fitting a scaler on the full dataset leaks test statistics into training.

### 30.29.2 Common estimators by task

| Task | Estimators |
|----|----|
| **Regression** | `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR` |
| **Classification** | `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `KNeighborsClassifier`, `GaussianNB` |
| **Clustering** | `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `GaussianMixture` |
| **Dimensionality reduction** | `PCA`, `TruncatedSVD`, `TSNE`, `NMF` |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder`, `SimpleImputer` |
| **Feature selection** | `SelectKBest`, `RFE`, `SelectFromModel` |

These map onto the chapters **Regression Models**, **Classification: Foundations**, **Support Vector Machines**, **Trees and Ensembles**, **Clustering**, and **Dimensionality Reduction**.

### 30.29.3 Pipelines

A **Pipeline** chains transformers and a final estimator into one object. It is not a convenience — it is the correct way to prevent data leakage, because `fit` propagates through every step using training data only, and cross-validation refits the whole chain on each fold.

``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)           # scaler fit on train only, then clf fit
pipe.predict(X_test)                 # test data scaled with train statistics
```

### 30.29.4 ColumnTransformer

Real datasets mix numeric and categorical columns needing different preprocessing. **ColumnTransformer** applies different transformers to different column subsets, then concatenates the results.

``` python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric = ["age", "income"]
categorical = ["city", "plan"]

pre = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

full = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
full.fit(X_train, y_train)           # every step leak-free and reproducible
```

### 30.29.5 Model selection and hyperparameter search

| Tool                       | Use                                           |
|----------------------------|-----------------------------------------------|
| `train_test_split`         | One-shot train/test split                     |
| `cross_val_score`          | K-fold cross-validated scores                 |
| `cross_validate`           | Multiple metrics + fit/score times            |
| `KFold`, `StratifiedKFold` | Fold generators (stratify for classification) |
| `GridSearchCV`             | Exhaustive hyperparameter grid                |
| `RandomizedSearchCV`       | Random search over distributions              |

``` python
from sklearn.model_selection import GridSearchCV, cross_val_score

scores = cross_val_score(full, X_train, y_train, cv=5, scoring="f1")
print(scores.mean(), scores.std())

grid = GridSearchCV(
    full,
    param_grid={"clf__C": [0.1, 1, 10], "clf__penalty": ["l2"]},
    cv=5, scoring="f1", n_jobs=-1,
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
best_model = grid.best_estimator_    # refit on all training data automatically
```

Note the `"clf__C"` naming: within a pipeline, address a step’s hyperparameter as `<step_name>__<param>`.

### 30.29.6 Metrics

| Task | Metrics | Import from `sklearn.metrics` |
|----|----|----|
| **Classification** | accuracy, precision, recall, F1, ROC-AUC | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `classification_report`, `confusion_matrix` |
| **Regression** | MAE, MSE, RMSE, R² | `mean_absolute_error`, `mean_squared_error`, `r2_score` |
| **Clustering** | silhouette, ARI | `silhouette_score`, `adjusted_rand_score` |

``` python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

preds = grid.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
proba = grid.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, proba))
```

Choosing the right metric is a modeling decision, not a formality — accuracy misleads on imbalanced data, where precision/recall/F1 or ROC-AUC tell the real story. See **Supervised Learning and Model Evaluation** for the full treatment.

## 30.30 PyTorch Quick Reference

**PyTorch** is a deep-learning framework built on **tensors** (GPU-capable arrays) and **autograd** (automatic differentiation). Its define-by-run style makes it the research default. The concepts here are developed in **Neural Network Fundamentals** and **Training Deep Neural Networks**.

``` python
import torch
import torch.nn as nn
```

### 30.30.1 Tensors

| Call | Result |
|----|----|
| `torch.tensor([1., 2., 3.])` | Tensor from data |
| `torch.zeros(2, 3)` / `torch.ones(2, 3)` | Filled tensors |
| `torch.randn(2, 3)` | Standard-normal random |
| `torch.arange(10)` | Range tensor |
| `t.shape`, `t.dtype`, `t.device` | Shape, type, location |
| `t.reshape(3, 4)` / `t.view(3, 4)` | Reshape |
| `t.to("cuda")` | Move to GPU |
| `t.numpy()` / `torch.from_numpy(a)` | NumPy interop (shares memory on CPU) |
| `t.item()` | Extract a Python scalar |

``` python
x = torch.randn(4, 3)                # a 4x3 tensor
print(x.shape, x.dtype, x.device)    # torch.Size([4, 3]) torch.float32 cpu
y = x @ torch.randn(3, 2)            # matrix product -> shape (4, 2)
```

### 30.30.2 Autograd

Set `requires_grad=True` and PyTorch records operations to build a computation graph; `.backward()` fills in `.grad` for every leaf.

``` python
w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
loss = (w * x - 1.0) ** 2            # a scalar function of w
loss.backward()                      # compute d(loss)/dw
print(w.grad)                        # tensor([12.])  -> 2*(w*x-1)*x = 2*5*3

with torch.no_grad():                # disable graph tracking (e.g. at inference)
    prediction = w * x
```

### 30.30.3 Defining a model with nn.Module

Subclass `nn.Module`, register layers in `__init__`, and define the forward pass in `forward`.

``` python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=20, hidden=64, out_dim=2)
print(sum(p.numel() for p in model.parameters()))  # parameter count
```

Common building blocks:

| Layer / op | Purpose |
|----|----|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.Conv2d(in_ch, out_ch, k)` | 2-D convolution |
| `nn.ReLU()`, `nn.GELU()` | Activations |
| `nn.Dropout(p)` | Regularization |
| `nn.BatchNorm1d(n)` / `nn.LayerNorm(n)` | Normalization |
| `nn.Embedding(vocab, dim)` | Lookup table for tokens |
| `nn.CrossEntropyLoss()` | Classification loss (expects logits) |
| `nn.MSELoss()` | Regression loss |

### 30.30.4 A minimal training loop

The canonical five steps per batch: zero gradients, forward, compute loss, backward, step.

``` python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Toy data
X = torch.randn(500, 20)
y = torch.randint(0, 2, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model = MLP(20, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    running = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()        # 1. clear old gradients
        logits = model(xb)           # 2. forward pass
        loss = criterion(logits, yb) # 3. loss
        loss.backward()              # 4. backprop
        optimizer.step()             # 5. update weights
        running += loss.item() * xb.size(0)
    print(f"epoch {epoch}: loss={running / len(loader.dataset):.4f}")
```

Evaluation disables gradients and dropout/batchnorm training behavior:

``` python
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
print("accuracy:", correct / len(loader.dataset))
```

### 30.30.5 Moving to the GPU

Move **both** the model and each batch to the same device.

``` python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(20, 64, 2).to(device)
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)   # batch must match model's device
    logits = model(xb)
    # ... training step as above
```

### 30.30.6 Saving and loading

Save the **state dict** (the learned parameters), not the whole object — it is portable and version-robust.

``` python
torch.save(model.state_dict(), "model.pt")

model = MLP(20, 64, 2)               # recreate the architecture first
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
```

## 30.31 Keras / TensorFlow Quick Reference

**Keras** is the high-level API of **TensorFlow**, prized for a concise `compile`/`fit` workflow. Two ways to build models: the **Sequential** API for simple stacks, and the **functional** API for branching or multi-input graphs.

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### 30.31.1 Sequential API

``` python
model = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(2, activation="softmax"),
])
model.summary()                      # print architecture and parameter counts
```

### 30.31.2 Functional API

For anything non-linear — multiple inputs, skip connections, shared layers:

``` python
inputs = keras.Input(shape=(20,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

### 30.31.3 Compile and fit

`compile` attaches the optimizer, loss, and metrics; `fit` runs the training loop for you.

``` python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",   # integer labels
    metrics=["accuracy"],
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)

loss, acc = model.evaluate(X_test, y_test)
preds = model.predict(X_test)        # probabilities -> argmax for the class
model.save("model.keras")            # save architecture + weights + optimizer
reloaded = keras.models.load_model("model.keras")
```

| Loss string                         | Use when                    |
|-------------------------------------|-----------------------------|
| `"sparse_categorical_crossentropy"` | Multi-class, integer labels |
| `"categorical_crossentropy"`        | Multi-class, one-hot labels |
| `"binary_crossentropy"`             | Binary classification       |
| `"mse"` / `"mae"`                   | Regression                  |

The `history.history` dict holds per-epoch loss and metric curves — plot it to diagnose over/underfitting, as discussed in **Training Deep Neural Networks**.

## 30.32 Hugging Face Transformers Quick Reference

The **Transformers** library gives one-line access to thousands of pretrained models. The architecture it implements is the subject of **Transformers and Attention**; here we cover only how to *use* it.

``` python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
```

### 30.32.1 The pipeline — the fast path

A `pipeline` bundles tokenizer, model, and post-processing behind a single callable.

``` python
clf = pipeline("sentiment-analysis")
print(clf("This appendix is genuinely useful."))
# -> [{'label': 'POSITIVE', 'score': 0.9998}]

ner = pipeline("ner", grouped_entities=True)
qa = pipeline("question-answering")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

| Task string              | Does                         |
|--------------------------|------------------------------|
| `"sentiment-analysis"`   | Text classification          |
| `"ner"`                  | Named-entity recognition     |
| `"question-answering"`   | Extractive QA over a context |
| `"summarization"`        | Abstractive summarization    |
| `"translation_xx_to_yy"` | Machine translation          |
| `"text-generation"`      | Autoregressive generation    |
| `"fill-mask"`            | Masked-token prediction      |
| `"feature-extraction"`   | Embeddings                   |

### 30.32.2 AutoTokenizer and AutoModel

For control beyond the pipeline, load the tokenizer and model explicitly. The `Auto*` classes pick the right concrete class from the model name.

``` python
name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(name)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(name)

inputs = tokenizer("A crisp sentence.", return_tensors="pt", truncation=True)
import torch
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)
print(probs)
```

| Auto class                           | Loads a model for            |
|--------------------------------------|------------------------------|
| `AutoModel`                          | Base encoder (hidden states) |
| `AutoModelForSequenceClassification` | Text classification          |
| `AutoModelForCausalLM`               | Text generation (GPT-style)  |
| `AutoModelForSeq2SeqLM`              | Encoder-decoder (T5, BART)   |
| `AutoModelForTokenClassification`    | NER / tagging                |
| `AutoModelForQuestionAnswering`      | Extractive QA                |

### 30.32.3 A generation snippet

``` python
name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

prompt = "Machine learning is"
inputs = tokenizer(prompt, return_tensors="pt")
output_ids = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

| Generation argument  | Effect                                     |
|----------------------|--------------------------------------------|
| `max_new_tokens`     | How many tokens to generate                |
| `do_sample`          | Sample (`True`) vs greedy/beam (`False`)   |
| `temperature`        | Higher = more random, lower = more focused |
| `top_k` / `top_p`    | Truncate the sampling distribution         |
| `num_beams`          | Beam search width (with `do_sample=False`) |
| `repetition_penalty` | Discourage repeats                         |

Push models to the GPU with `model.to("cuda")` and matching `inputs.to("cuda")`, exactly as in the PyTorch section.

## 30.33 LLM API Usage Quick Reference

Frontier models are most often used through a hosted API rather than run locally. The pattern below is the **Anthropic Claude** API via the official Python SDK. The higher-level design questions — prompting, tools, retrieval — are covered in **LLMs in Practice**, **Retrieval-Augmented Generation**, and **Agentic AI**.

``` bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."   # never hardcode keys in source
```

### 30.33.1 A minimal call

The core endpoint is the **Messages API**. You send a list of messages (each with a `role` and `content`) and receive a structured response whose text lives in `response.content`.

``` python
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the env

response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain gradient descent in two sentences."},
    ],
)

# content is a list of typed blocks; extract the text block(s)
for block in response.content:
    if block.type == "text":
        print(block.text)
```

### 30.33.2 The messages format

The conversation is a list of alternating turns. A `system` prompt (top-level, not a message) sets behavior; `user` and `assistant` roles carry the dialogue. The API is **stateless** — send the full history each time.

``` python
response = client.messages.create(
    model="claude-sonnet-4-6",           # a faster, lower-cost model
    max_tokens=1024,
    system="You are a concise ML tutor. Prefer examples over prose.",
    messages=[
        {"role": "user", "content": "What is overfitting?"},
        {"role": "assistant", "content": "When a model memorizes training noise..."},
        {"role": "user", "content": "How do I detect it?"},   # multi-turn context
    ],
)
```

| Field | Meaning |
|----|----|
| `model` | Model ID, e.g. `"claude-opus-4-8"` or `"claude-sonnet-4-6"` |
| `max_tokens` | Upper bound on tokens generated in the reply |
| `system` | Top-level instruction string (not part of `messages`) |
| `messages` | List of `{"role": "user"\|"assistant", "content": ...}` |
| `response.content` | List of content blocks; read `.text` on text blocks |
| `response.usage` | Input/output token counts for cost tracking |
| `response.stop_reason` | Why generation stopped (`end_turn`, `max_tokens`, …) |

Two practical notes. First, on recent Claude models you enable step-by-step reasoning with **adaptive thinking** — pass `thinking={"type": "adaptive"}` and let the model decide how much to reason, rather than tuning a fixed token budget. Second, for long or high-`max_tokens` requests, **stream** the response so you do not hit request timeouts:

``` python
with client.messages.stream(
    model="claude-opus-4-8",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Draft a short lesson on backprop."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()   # full structured message when done
```

Always read a provider’s current documentation before building against it — model IDs, parameters, and capabilities change, and the SDK’s own help is the authority.

## 30.34 Git for ML and Experiment Tracking

Version control for ML spans three artifacts that plain git handles unevenly: **code** (git’s home turf), **data and models** (too large for git), and **experiments** (runs, metrics, hyperparameters — not files at all).

### 30.34.1 Git essentials

| Command                        | Does                                       |
|--------------------------------|--------------------------------------------|
| `git init` / `git clone <url>` | Start / copy a repo                        |
| `git status`                   | Show working-tree changes                  |
| `git add -p`                   | Stage changes interactively (hunk by hunk) |
| `git commit -m "msg"`          | Record a snapshot                          |
| `git switch -c feature`        | Create and switch to a branch              |
| `git merge feature`            | Merge a branch in                          |
| `git log --oneline --graph`    | Compact history view                       |
| `git diff`                     | Show unstaged changes                      |
| `git stash`                    | Shelve changes temporarily                 |

A `.gitignore` for ML keeps the repo clean and safe:

``` gitignore
# Never commit these
.venv/
__pycache__/
.env                 # secrets and API keys
*.ckpt               # large model checkpoints
data/raw/            # large datasets -> track with DVC instead
.ipynb_checkpoints/
wandb/               # local tracking logs
```

**Never commit secrets.** API keys, credentials, and tokens belong in environment variables or a secrets manager, referenced from `.env` files that are git-ignored. A leaked key in git history is compromised even after you delete it.

### 30.34.2 Versioning data and models

Git stores text diffs efficiently but chokes on large binaries. Two common solutions:

| Tool | Approach |
|----|----|
| **DVC** | Stores a small text pointer in git; the data lives in remote object storage (S3, GCS). `dvc add data/`, `dvc push`. |
| **Git LFS** | Replaces large files with pointers, stored on an LFS server. Good for moderate binaries. |

``` bash
# DVC: version a dataset without bloating git
dvc init
dvc add data/train.parquet          # creates data/train.parquet.dvc (tiny, in git)
git add data/train.parquet.dvc .gitignore
dvc remote add -d storage s3://my-bucket/dvcstore
dvc push                            # upload the actual data to remote storage
```

### 30.34.3 Experiment tracking

An experiment tracker logs each run’s hyperparameters, metrics, and artifacts so you can compare runs and reproduce the best one. The chapter **MLOps and Productionizing Models** covers the discipline; the tools:

| Tool | Character |
|----|----|
| **MLflow** | Open-source, self-hostable; tracking + model registry |
| **Weights & Biases** | Hosted, rich dashboards, collaboration |
| **TensorBoard** | Lightweight, tightly integrated with TF/PyTorch |

``` python
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 1e-3)
    mlflow.log_param("batch_size", 32)
    for epoch, loss in enumerate(losses):
        mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("test_accuracy", 0.91)
    mlflow.log_artifact("model.pt")     # attach the trained model
```

The minimum viable practice: log every run’s **config, metrics, and the code commit hash** together, so any result can be traced back to the exact code and settings that produced it.

## 30.35 Troubleshooting and Common Errors

The errors below account for a large share of day-to-day friction. The pattern to internalize: read the *last* line of a traceback first (the actual exception), then work upward to find your code.

| Symptom / error | Likely cause | Fix |
|----|----|----|
| `ModuleNotFoundError: No module named X` | Package not installed in the active env | Activate the right env; `pip install X`; check `which python` |
| `ImportError` after install | Installed into a different interpreter | Verify `pip` and `python` point to the same env |
| NumPy `ValueError: operands could not be broadcast` | Shape mismatch | Print `.shape` of each operand; add/align axes |
| `ValueError: could not convert string to float` | Non-numeric data reaching a numeric model | Encode categoricals; clean the column |
| pandas `SettingWithCopyWarning` | Chained assignment on a slice | Use `.loc[rows, cols] = ...` on the original frame |
| pandas `KeyError: 'col'` | Column name typo / whitespace | Inspect `df.columns`; `.str.strip()` headers |
| sklearn `NotFittedError` | Called `predict` before `fit` | Fit the estimator (or pipeline) first |
| sklearn shape error in `predict` | Test features differ from train | Use one Pipeline/ColumnTransformer for both |
| Suspiciously perfect validation score | **Data leakage** | Fit preprocessing inside CV; never fit on test |
| PyTorch `Expected all tensors on same device` | Model on GPU, batch on CPU (or vice versa) | Move both with `.to(device)` |
| PyTorch loss is `nan` | Exploding gradients / bad LR / log(0) | Lower LR; clip gradients; check for zeros in logs |
| PyTorch loss not decreasing | `optimizer.zero_grad()` missing; wrong LR | Confirm the five-step loop; sweep the LR |
| PyTorch `CUDA out of memory` | Batch too large / graph retained | Reduce batch size; use `torch.no_grad()` at eval |
| GPU not detected | Driver/toolkit/framework mismatch | Check `nvidia-smi`; install the matching build |
| Transformers download stalls / auth error | Network / gated model | Set cache dir; `huggingface-cli login` for gated repos |
| LLM API `rate_limit` / 429 | Too many requests | Back off and retry with exponential delay |
| LLM output truncated | `max_tokens` too low | Raise `max_tokens`; stream long outputs |
| Results differ every run | Unseeded randomness | Call a `set_seeds` helper at startup |

## 30.36 Keyboard-Free Recipes

Short, end-to-end snippets you can adapt without retyping the boilerplate. Each is deliberately complete.

### 30.36.1 Recipe 1 — Train and evaluate a classifier (scikit-learn)

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, preds))
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
```

### 30.36.2 Recipe 2 — Cross-validated hyperparameter search

``` python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

search = RandomizedSearchCV(
    model,
    param_distributions={
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_leaf": randint(1, 10),
    },
    n_iter=20, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1,
)
search.fit(X_train, y_train)
print("best AUC:", round(search.best_score_, 4))
print("best params:", search.best_params_)
best = search.best_estimator_        # refit on all training data
```

### 30.36.3 Recipe 3 — Save and load a scikit-learn model

``` python
import joblib

joblib.dump(best, "model.joblib")            # persist the whole pipeline
reloaded = joblib.load("model.joblib")        # reload anywhere with the same libs
assert (reloaded.predict(X_test) == best.predict(X_test)).all()
```

Persisting the **whole pipeline** (not just the estimator) means preprocessing travels with the model — the reloaded object accepts raw features and handles scaling and encoding itself.

### 30.36.4 Recipe 4 — Train a PyTorch model and checkpoint it

``` python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.tensor(X_train.values, dtype=torch.float32)
yt = torch.tensor(y_train.values, dtype=torch.long)
loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)

net = nn.Sequential(
    nn.Linear(Xt.shape[1], 64), nn.ReLU(),
    nn.Linear(64, 2),
).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(15):
    net.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(net(xb), yb)
        loss.backward()
        opt.step()

torch.save(net.state_dict(), "net.pt")   # checkpoint the parameters
```

### 30.36.5 Recipe 5 — Load a checkpoint and run inference

``` python
net = nn.Sequential(
    nn.Linear(Xt.shape[1], 64), nn.ReLU(),
    nn.Linear(64, 2),
).to(device)
net.load_state_dict(torch.load("net.pt", map_location=device))
net.eval()

Xtest = torch.tensor(X_test.values, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = net(Xtest).argmax(dim=1).cpu().numpy()
```

### 30.36.6 Recipe 6 — Zero-shot classification with a Transformers pipeline

``` python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "The GPU ran out of memory during the third epoch.",
    candidate_labels=["hardware", "billing", "user error", "networking"],
)
print(result["labels"][0], round(result["scores"][0], 3))
# -> hardware 0.9xx
```

## 30.37 Keeping Tooling Current

Everything in this appendix is a snapshot. The libraries evolve, APIs shift, and the model IDs and parameters in the LLM section will change faster than anything else. That is not a flaw to work around — it is the nature of a fast-moving field, and it makes one habit essential: **trust the source, not the snapshot.**

In practice this means a few small disciplines. Read release notes when you upgrade — major versions of NumPy, pandas, and scikit-learn have changed defaults and removed deprecated calls, and the notes tell you exactly what to fix. Prefer a library’s own help (`help()`, `?` in IPython, the official docs) over any printed reference, including this one, when the two disagree. Pin versions in a lockfile so an upstream change never breaks a working project silently, and upgrade deliberately, on your own schedule, with tests to catch regressions. For anything touching a hosted API, check the provider’s current documentation before you build — assume the model catalog and parameters have moved since you last looked.

The goal of a cheat sheet is to make the common case fast, not to freeze the field in amber. Use these pages to move quickly; use the primary sources to stay correct. When they conflict, the primary source wins.

---

[← Appendix A: Mathematical Notation and Reference](ch31-appendix-a-math-notation.md) · [Contents](../README.md#contents) · [Appendix C: Datasets, Projects, and Further Reading →](ch33-appendix-c-datasets-resources.md)
