# ProjectedConeOracle.jl

**Oracle-driven double-description algorithm for projected polyhedral cones.**

Part of the [Kroneckera/bunkbed](https://github.com/Kroneckera/bunkbed) repository.

---

## Overview

Given a polyhedral cone $C = \lbrace x \in \mathbb{R}^n : Ax \geq 0\rbrace$ and a set of projection indices $J \subseteq \lbrace 1, \ldots, n\rbrace$, ProjectedConeOracle.jl computes the **projected cone**

$$D = \pi_J(C) = \lbrace y \in \mathbb{R}^d : \exists z \text{ such that } A\begin{bmatrix} y \\ z \end{bmatrix} \geq 0\rbrace ,$$

where $d = \lvert J\rvert$ and the coordinates of $y$ correspond to the indices in $J$.

The algorithm returns both the **extreme rays** and the **facet normals** (support hyperplanes) of $D$, providing a complete double description of the projected cone. It automatically detects and exploits lower-dimensional structure via subspace reduction, so that if $D$ lives in a proper subspace of $\mathbb{R}^d$, the internal computation works in the minimal dimension.

## Mathematical Background

### The projection problem

Let $A \in \mathbb{R}^{m \times n}$ define a polyhedral cone $C = \lbrace x \in \mathbb{R}^n : Ax \geq 0\rbrace$, and let $J = \lbrace j_1, \ldots, j_d\rbrace$ be a subset of coordinate indices. The projection $D = \pi_J(C)$ is the set of all $y \in \mathbb{R}^d$ that extend to some feasible $x \in C$. Equivalently, $D$ is the image of $C$ under the linear map that extracts the $J$-coordinates.

The cone $D$ is itself polyhedral and can be described in two dual ways:

- **V-description (generators):** $D = \text{cone}(r_1, \ldots, r_k)$, the conic hull of its extreme rays.
- **H-description (inequalities):** $D = \lbrace y : h_i \cdot y \geq 0, i = 1, \ldots, f\rbrace$, the intersection of half-spaces defined by its facet normals.

### Motivation: universal inequalities from projected cones

In percolation theory and probabilistic combinatorics, one studies the space of feasible partition-function values. The feasible-potentials LP defines a cone $C$ in a high-dimensional space of auxiliary variables $\phi$, and the quantities of interest -- for instance, the symmetric polynomial coordinates $S_{p,q} = \mu(p)\mu(q)$ -- correspond to a low-dimensional projection. The extreme rays of $D$ are exactly the **universal inequalities** satisfied by partition probabilities: every valid probability assignment obeys these inequalities regardless of the underlying graph or parameters.

### Why oracle-based projection

Classical approaches to polyhedral projection -- Fourier-Motzkin elimination, block elimination, and their variants -- suffer from doubly exponential blowup in the number of intermediate inequalities. This makes them impractical even for moderate ambient dimension $n$. The oracle-based approach in ProjectedConeOracle.jl avoids this by never constructing intermediate representations. Instead, it works entirely in the $d$-dimensional projection space, using LP solves as a black-box oracle to access the high-dimensional cone $C$, and calls Normaliz only on the low-dimensional inner approximation. This makes the algorithm practical whenever $d$ is moderate (say, $d \leq 20$), regardless of $n$.

## Algorithm

The algorithm proceeds in five phases:

### Step 1: Dual normalization

Find a vector $c \in \text{relint}(D^*)$ -- the relative interior of the dual cone -- such that the slice $\lbrace y \in D : c \cdot y = 1\rbrace$ is a bounded polytope. This is computed by solving a max-min LP over the dual constraints, yielding a strictly feasible dual vector $\lambda$, from which $c = A_J^T \lambda$.

### Step 2: Seed rays

Generate initial extreme rays by solving $2d$ LP problems: for each coordinate axis $\pm e_t$, minimize $\pm e_t \cdot y$ over the normalization slice $\lbrace y \in D : c \cdot y = 1\rbrace$. Each optimizer gives a candidate extreme ray of $D$.

### Step 3: Subspace detection

Construct an orthonormal basis for the affine span of the discovered rays. If the rays span only a $k$-dimensional subspace of $\mathbb{R}^d$ with $k < d$, all subsequent computation is performed in $\mathbb{R}^k$. The basis is expanded dynamically as new rays outside the current span are discovered.

### Step 4: Main loop (separation--enumeration iteration)

Repeat until convergence or the iteration limit is reached:

1. **Rationalize:** Convert the current floating-point rays to primitive integer vectors via coordinate-wise rational approximation (controlled by `rat_tol` and `max_den`).
2. **Facet enumeration:** Pass the integer ray matrix to Normaliz to compute support hyperplanes of the current inner approximation $\hat{D} \subseteq D$.
3. **LP separation:** For each candidate facet normal $h$, solve the separation LP $\min\lbrace h \cdot y : y \in D, c \cdot y = 1\rbrace$.
   - If $h \cdot y^* < -\varepsilon$ (violation found), then $y^*$ is a new extreme ray of $D$ not yet in $\hat{D}$. Add it and restart.
   - If $h \cdot y^* \geq -\varepsilon$ for all facets, the inner approximation is exact and the algorithm has converged.
4. **Two-phase verification:** Each separation LP is solved first with an interior-point method (fast) and then re-solved with simplex (numerically reliable basic solution) when a violation is detected.

### Step 5: Lift and return

Project the final ray and facet matrices from the working subspace back to the ambient $d$-dimensional space. Deduplicate and canonicalize all vectors.

## Quick Start

```julia
using ProjectedConeOracle
using SparseArrays

# Example 1: The 2D positive orthant (identity matrix)
A = sparse(Float64[1 0; 0 1])
result = projected_cone(A, [1, 2])

result.rays     # [[0.0, 1.0], [1.0, 0.0]]
result.facets   # [[0.0, 1.0], [1.0, 0.0]]
result.converged  # true

# Example 2: Project the 3D positive orthant onto coordinates 1 and 3
A = sparse(Matrix{Float64}(I, 3, 3))
result = projected_cone(A, [1, 3])

result.rays     # [[0.0, 1.0], [1.0, 0.0]]
result.facets   # [[0.0, 1.0], [1.0, 0.0]]

# Example 3: A cone with a single diagonal ray
A = sparse([
    1.0 -1.0
   -1.0  1.0
    1.0  1.0
])
result = projected_cone(A, [1, 2])

result.rays     # [[1.0, 1.0]]   (the ray y1 = y2)
result.facets   # [[1.0, 1.0]]   (single supporting hyperplane)
```

## Installation

ProjectedConeOracle.jl is not yet registered in the Julia General registry. Install it by developing from a local clone of the repository:

```julia
using Pkg
Pkg.develop(path="/path/to/bunkbed/ProjectedConeOracle")
```

### External dependencies

- **Normaliz** (via `normaliz_jll`): Used for exact facet enumeration. Installed automatically as a JLL artifact.
- **HiGHS** (via `HiGHS.jl`): Default LP solver. Installed automatically as a dependency.
- **Gurobi** (optional): Load with `using Gurobi` before constructing a `GurobiAdapter`. Requires a valid Gurobi license.
- **GLPK** (optional): Load with `using GLPK` before constructing a `GLPKAdapter`.

## API Reference

### Entry points

#### `projected_cone(A, J; config, solver, adapter, backend)`

Convenience wrapper with keyword-only arguments. `solver` and `adapter` are synonyms.

```julia
result = projected_cone(A, J;
    config  = OracleConfig(),
    solver  = HiGHSAdapter(),      # or GurobiAdapter(), GLPKAdapter()
    backend = NormalizCLIBackend(),
)
```

#### `projected_cone_oracle(A, J, config, adapter; backend)`

Full-signature entry point with positional arguments.

```julia
result = projected_cone_oracle(A, J, config, adapter; backend=NormalizCLIBackend())
```

**Arguments:**

| Parameter | Type | Description |
|-----------|------|-------------|
| A | AbstractMatrix{<:Real} | Constraint matrix ($m \times n$), defining $C = \lbrace x : Ax \geq 0\rbrace$ |
| J | AbstractVector{<:Integer} | Projection indices (1-based), $\lvert J\rvert = d$ |
| config | OracleConfig | Algorithm configuration |
| adapter | AbstractSolverAdapter | LP solver backend |
| backend | AbstractNormalizBackend | Normaliz backend (keyword, default: NormalizCLIBackend()) |

### `ProjectedConeResult`

Returned by both entry points.

| Field | Type | Description |
|-------|------|-------------|
| rays | Vector{Vector{Float64}} | Extreme rays of the projected cone $D$ |
| facets | Vector{Vector{Float64}} | Facet normals of $D$; each $h$ satisfies $h \cdot y \geq 0$ for all $y \in D$ |
| iterations | Int | Number of separation--enumeration iterations |
| lp_calls | Int | Total LP solves (including two-phase verification) |
| converged | Bool | true if the algorithm terminated with a provably complete description |

### `OracleConfig`

All tolerances are derived from a single master tolerance `base_eps` by default.

| Parameter | Default | Description |
|-----------|---------|-------------|
| base_eps | 1e-9 | Master tolerance; most other tolerances scale from this |
| canonical_tol | base_eps * 1e-3 | Near-zero threshold for canonicalization |
| quantization_eps | base_eps * 10 | Hash-key rounding for ray/facet deduplication |
| pointedness_tol | base_eps * 10 | Tolerance for pointedness checks |
| homogeneity_tol | base_eps * 100 | Tolerance for homogeneity checks |
| violation_tol | base_eps * 1000 | LP separation threshold; violation below this triggers a new ray |
| seed_tol | violation_tol | Tolerance for deterministic seeding and subspace expansion |
| cy_normalize_tol | 1e-12 | Minimum $\|c \cdot y\|$ for normalization to succeed |
| rat_tol | 1e-10 | Initial tolerance for rationalize |
| max_den | 10_000_000 | Maximum denominator in rational approximation |
| rat_tol_ceiling | 1e-4 | Maximum tolerance after adaptive doubling |
| max_iterations | 10_000 | Iteration limit (alias: max_rounds) |
| prune_every | 100 | Prune duplicate rays every $N$ iterations (0 = never) |
| shuffle_facets | true | Randomly permute facet processing order each iteration |
| normaliz_threads | 0 | Thread count for Normaliz (0 = use all available CPUs) |
| normaliz_verbose | false | Print Normaliz output to log_io |
| normaliz_keep_order | true | Pass -k flag to Normaliz (preserves generator order) |
| normaliz_algorithm | :projection_float | One of :none, :projection, :projection_float |
| checkpoint_file | nothing | Path for resumable checkpoints (Julia serialization) |
| checkpoint_reorthogonalize | true | Re-orthogonalize the subspace basis when loading checkpoints |
| verbose | false | Print iteration-level progress to log_io |
| log_io | stderr | Output stream for verbose logging |

### Solver adapters

#### `HiGHSAdapter(; solver, primal_feasibility_tolerance, dual_feasibility_tolerance, time_limit)`

Default LP backend. Always available.

| Parameter | Default | Description |
|-----------|---------|-------------|
| solver | "ipm" | "ipm" (interior-point) or "simplex" |
| primal_feasibility_tolerance | 1e-9 | Primal feasibility tolerance |
| dual_feasibility_tolerance | 1e-9 | Dual feasibility tolerance |
| time_limit | Inf | Per-solve time limit in seconds |

#### `GurobiAdapter(; method, crossover, presolve, feasibility_tol, optimality_tol, numeric_focus, time_limit)`

Requires `using Gurobi` (loaded via package extension).

| Parameter | Default | Description |
|-----------|---------|-------------|
| method | 1 | -1=auto, 0=primal simplex, 1=dual simplex, 2=barrier |
| crossover | -1 | -1=auto, 0=disable, 1=primal, 2=dual |
| presolve | 0 | Gurobi presolve (0=off recommended for this package) |
| feasibility_tol | 1e-9 | Feasibility tolerance |
| optimality_tol | 1e-9 | Optimality tolerance |
| numeric_focus | 0 | 0--3; higher trades speed for reliability |
| time_limit | Inf | Per-solve time limit in seconds |

#### `GLPKAdapter(; method, tol_bnd, tol_dj, time_limit)`

Requires `using GLPK` (loaded via package extension).

| Parameter | Default | Description |
|-----------|---------|-------------|
| method | :InteriorPoint | :InteriorPoint or :Simplex |
| tol_bnd | 1e-9 | Bound tolerance |
| tol_dj | 1e-9 | Dual feasibility tolerance |
| time_limit | Inf | Per-solve time limit in seconds |

### Normaliz backend

#### `NormalizCLIBackend()`

The default (and only production) backend. Invokes the `normaliz` binary from `normaliz_jll`, writes `.in` files, and parses `.sup`/`.ext`/`.out` output.

#### `compute_cone(backend, generators, config; input_type) -> NormalizResult`

Low-level interface to Normaliz. Computes support hyperplanes, extreme rays, and lineality space from a generator matrix.

#### `NormalizResult`

| Field | Type | Description |
|-------|------|-------------|
| support_hyperplanes | Matrix{BigInt} | Facet normals (rows) |
| extreme_rays | Matrix{BigInt} | Extreme rays (rows) |
| lineality_space | Matrix{BigInt} | Lineality basis (rows; empty if pointed) |

### Separation oracle

#### `SeparationOracle(A, J; adapter, c, config)`

Constructs the LP model for separation queries against $D = \pi_J(\lbrace x : Ax \geq 0\rbrace )$.

#### `solve_oracle!(oracle, h; canonicalize) -> (obj, y)`

Single-phase LP solve: minimize $h \cdot y$ subject to $y \in D$, $c \cdot y = 1$.

#### `solve_verified!(oracle, h; canonicalize, verify_tol) -> (obj, y, lp_count)`

Two-phase LP solve: barrier first (fast), then simplex (numerically reliable). Returns `lp_count` (1 or 2). Skips the verification phase if the objective exceeds `verify_tol`.

### Subspace utilities

#### `SubspaceBasis`

Orthonormal basis for a working subspace of $\mathbb{R}^d$. Manages dynamic dimension expansion.

- `project(sb, y) -> Vector{Float64}` -- Project $y$ to reduced coordinates: $z = B^T y$.
- `lift(sb, z) -> Vector{Float64}` -- Lift $z$ to ambient coordinates: $y = Bz$.
- `maybe_expand!(sb, y, config) -> Bool` -- Expand the basis if $y$ has a significant component perpendicular to the current span. Returns `true` if the basis was expanded.
- `cy_normalize(z, c, config) -> Vector{Float64}` -- Normalize $z$ so that $c \cdot z = 1$.

### Canonicalization

- `canonical_ray(r, config) -> Vector{Float64}` -- Normalize ray by $\ell^\infty$-norm.
- `canonical_normal(h, config) -> Vector{Float64}` -- Normalize facet normal by $\ell^\infty$-norm.
- `hash_key_ray(r, config) -> Tuple{Vararg{Int}}` -- Quantized integer tuple for ray deduplication.
- `hash_key_normal(h, config) -> Tuple{Vararg{Int}}` -- Quantized integer tuple for normal deduplication.

### Rationalization

- `rationalize_ray_coordwise(r, config) -> Vector{BigInt}` -- Convert a floating-point ray to a primitive integer vector via coordinate-wise rational approximation.
- `gcd_vec(z) -> Integer` -- GCD of all elements in an integer vector.

### Facet enumeration

- `enumerate_facets(rays, config; backend) -> Matrix{BigInt}` -- Compute facet normals of the cone generated by the rows of `rays`. Errors if the cone is not pointed.
- `validate_facet(normal, rays, config) -> Bool` -- Check that `normal` is valid ($h \cdot r \geq 0$ for all generators $r$).

### Normalization

- `find_normalization_vector(A, J, config, adapter) -> (c, seeds)` -- Compute $c \in \text{relint}(D^*)$ and initial seed rays.
- `find_dual_normalization_vector(A, J; adapter, config) -> (c, seeds)` -- Keyword-argument wrapper (backward compatibility).

### Checkpointing

- `save_checkpoint(file, R, iter, lp_calls, B, config)` -- Serialize oracle state for resumable computation.
- `load_checkpoint(file, config) -> (R, seen_rays, iter, lp_calls, B)` -- Deserialize a checkpoint.

## Solver Backends

| Solver | Package | License | Two-phase | Notes |
|--------|---------|---------|-----------|-------|
| **HiGHS** | HiGHS.jl | MIT | Yes (IPM + simplex) | Default backend. Open-source, no license required. |
| **Gurobi** | Gurobi.jl | Commercial | Yes (primal simplex + dual simplex) | Fastest for large-scale problems. Requires a valid Gurobi license. Load with using Gurobi. |
| **GLPK** | GLPK.jl | GPL | Yes (interior-point + simplex) | Open-source alternative. Load with using GLPK. |

All three adapters support the two-phase verification protocol: fast interior-point solve followed by simplex re-solve for a numerically reliable basic solution. Automatic recovery from `NUMERICAL_ERROR` is built in for all backends.

To use a non-default solver:

```julia
using Gurobi  # triggers the GurobiExt package extension
result = projected_cone(A, J; solver=GurobiAdapter(numeric_focus=1))
```

## Checkpointing

For long-running computations, enable checkpointing to allow resumable runs:

```julia
config = OracleConfig(
    checkpoint_file = "my_cone.checkpoint.jls",
    verbose = true,
)
result = projected_cone(A, J; config=config)
```

If the computation is interrupted, re-running the same call will resume from the last checkpoint. The checkpoint file stores:

- All discovered rays (rationalized to `BigInt` for exact reproducibility).
- The current subspace basis (re-orthogonalized on load by default).
- Iteration and LP call counts.

Checkpoints use Julia's `Serialization` format and are specific to the Julia version.

## Testing

Run the full test suite:

```bash
cd ProjectedConeOracle
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite covers canonicalization, rationalization, subspace management, checkpoint round-tripping, solver adapter configuration, Normaliz CLI integration, facet enumeration, normalization vector computation, adversarial inputs, and end-to-end projection algorithm correctness.

## Architecture

```
src/
  ProjectedConeOracle.jl   Module definition, includes, and exports
  types.jl                 ProjectedConeResult, CheckpointPayload, HashKey alias
  config.jl                OracleConfig struct with validated defaults
  solver_adapter.jl        AbstractSolverAdapter, HiGHSAdapter (+ Gurobi/GLPK stubs)
  SeparationOracle.jl      LP-based separation oracle (JuMP model construction + solve)
  ProjectionAlgorithm.jl   Main loop: projected_cone_oracle, projected_cone
  dual_normalization.jl    find_normalization_vector, seed ray generation
  subspace.jl              SubspaceBasis: project, lift, maybe_expand!, cy_normalize
  normaliz_cli.jl          NormalizCLIBackend: write input, run binary, parse output
  canonicalization.jl       canonical_ray, canonical_normal, hash_key_ray, hash_key_normal
  rationalization.jl       rationalize_ray_coordwise, gcd_vec
  facet_enumeration.jl     enumerate_facets, validate_facet
  checkpoint.jl            save_checkpoint, load_checkpoint

ext/
  GurobiExt.jl             Gurobi solver adapter (package extension)
  GLPKExt.jl               GLPK solver adapter (package extension)

test/
  runtests.jl              Test entry point
  test_*.jl                Unit and integration tests for each module
```

## Citation

If you use ProjectedConeOracle.jl in your research, please cite:

```bibtex
@software{projected_cone_oracle_2026,
  author = {Gladkov, Nikita and Zimin, Aleksandr},
  title  = {{ProjectedConeOracle.jl}: Oracle-driven double-description
            algorithm for projected polyhedral cones},
  year   = {2026},
  url    = {https://github.com/Kroneckera/bunkbed},
}
```

## License

See the repository root for license information.
