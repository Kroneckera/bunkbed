# Inequality Enumeration Pipeline

Production pipeline for enumerating universal bond-percolation inequalities
on connectivity partition probabilities. Connects the decision-tree feasibility
oracle (D1) to the projected cone oracle (D2).

## Quick Start

```julia
using PercolationOracle
using ProjectedConeOracle

# Enumerate all inequalities for n=3 with m=4 decision trees
result = enumerate_all_inequalities(3, 4)

# Print the inequalities
for ineq in result.formatted_inequalities
    println(ineq)
end
```

Output for m=4:
```
mu(123)*mu(1|2|3) - mu(12|3)*mu(13|2) - mu(12|3)*mu(1|23) - mu(13|2)*mu(1|23) >= 0
mu(123)*mu(12|3) + mu(123)*mu(13|2) + mu(123)*mu(1|23) - mu(123)*mu(1|2|3) + 2*mu(12|3)*mu(13|2) + mu(12|3)*mu(1|23) + mu(13|2)*mu(1|23) + mu(1|23)*mu(1|2|3) >= 0
mu(123)^2 >= 0
...
```

## API Reference

### `enumerate_all_inequalities(n_obs, m; kwargs...) -> InequalityEnumerationResult`

Run the full D1 -> D2 pipeline to enumerate all universal inequalities.

**Arguments:**
- `n_obs::Int` -- number of terminals. Currently only `n_obs=3` is supported.
- `m::Int` -- number of decision trees (1 to 4). Controls the prefix length
  from the paper family `(T0, T1, T2, T3)`.

**Keyword arguments:**
- `solver::AbstractSolverAdapter = HiGHSAdapter()` -- LP solver backend.
  `HiGHSAdapter()` is the current validated default for the full `m=4` run; explicit
  `GurobiAdapter(...)` requests are no longer silently retried with HiGHS if Gurobi fails.
- `verbose::Bool = true` -- print iteration-level progress to stderr.
- `max_iterations::Int = 5000` -- maximum oracle iterations before stopping.

**Returns:** `InequalityEnumerationResult` (see below).

**Examples:**

```julia
# Basic usage with HiGHS (free, no license needed)
result = enumerate_all_inequalities(3, 2)
@assert result.projection_result.converged
@assert length(result.rays) == 15  # m=2: simplicial cone

# Optional explicit solver selection
result = enumerate_all_inequalities(3, 4;
    solver=HiGHSAdapter(solver="simplex"),
    max_iterations=5000)
@assert length(result.rays) == 17  # m=4: 17 inequalities
```

### `InequalityEnumerationResult`

Result struct with fields:

| Field | Type | Description |
|-------|------|-------------|
| `facet_normals` | `Vector{Vector{Float64}}` | Facet normals / meta-constraints of the projected cone in S-coordinates |
| `inequalities` | alias | Backward-compatible alias for `facet_normals` |
| `rays` | `Vector{Vector{Float64}}` | Extreme rays = the valid universal inequalities |
| `labels` | `Vector{String}` | Symbolic labels for S-coordinates (e.g., `"mu(123)^2"`) |
| `formatted_inequalities` | `Vector{String}` | Human-readable `"... >= 0"` strings |
| `projection_result` | `ProjectedConeResult` | Raw D2 result (rays, facets, iterations, lp_calls, converged) |
| `timing` | `Float64` | Wall-clock seconds for the D2 stage |

**Interpreting results:**

The `rays` are the actual inequalities. Each ray `r` in R^15 defines:
```
sum_{p <= q} r[idx(p,q)] * mu(p) * mu(q) >= 0
```
where `mu(p)` is the probability that the connectivity partition equals `p`.

For `n=3`, the public labels use the paper order `(123, 1|2|3, 1|23, 12|3, 13|2)`. The `labels` vector maps each coordinate to its monomial:
```
Index 1:  mu(123)^2           (diagonal: p=123, q=123)
Index 2:  mu(1|2|3)^2         (diagonal)
...
Index 6:  mu(123)*mu(1|2|3)   (off-diagonal: p=123, q=1|2|3)
...
Index 15: mu(12|3)*mu(13|2)   (off-diagonal: p=12|3, q=13|2)
```

The `formatted_inequalities` express each ray as a readable polynomial inequality.

## Pipeline Stages

### Stage 1: Tuple acquisition

Loads the included legacy m=4 feasible-tuple archive from `data/valid_partition_tuples_nobs3_notebook.jls`,
normalizes it on load to the paper tree order `(T0, T1, T2, T3)`, then projects to the target `m` by taking the first `m` tree pairs. Deduplicates via `Set`.

```
m=1: 25 unique tuples
m=2: 139 unique tuples
m=3: 570 unique tuples
m=4: 1265 unique tuples
```

### Stage 2: Base constraint matrix M_F

Builds a sparse matrix M_F of shape `(|F|, m * bell^2)` where `bell = Bell(n) = 5`
for n=3. Each row has exactly m nonzero entries, one per tree, at column
`phi_index(k, p, pbar, n_obs)`.

### Stage 3: Symmetric extension

Appends 15 symmetric polynomial variables S_{p,q} (5 diagonal + 10 off-diagonal)
via equality constraints encoded as opposing inequality pairs. The extended matrix
has dimensions `(|F| + 30, m*25 + 15)`.

| Variable | Definition | Monomial meaning |
|----------|-----------|------------------|
| S_{p,p} | sum_k phi_k(p,p) | mu(p)^2 |
| S_{p,q} (p<q) | sum_k [phi_k(p,q) + phi_k(q,p)] | mu(p)*mu(q) |

### Stage 4: Projected cone oracle

Passes the extended matrix and projection indices to `ProjectedConeOracle.projected_cone_oracle`.
The oracle discovers all extreme rays and facets of the projected cone D = pi_J(C)
where C = {x : Ax >= 0} and J indexes the 15 S-variables.

### Stage 5: Result formatting

Converts each extreme ray to a human-readable polynomial inequality by:
1. Rationalizing coefficients to coprime integers via `canonicalize_integer_ray`
2. Mapping coordinate indices to monomial labels
3. Formatting as `"coeff1*label1 + coeff2*label2 + ... >= 0"`

## Known Results

### m=2: 15 inequalities (simplicial cone)

All 15 inequalities are either single-variable nonnegativity or simple sums:
- 5 diagonal: `mu(p)^2 >= 0`
- 8 off-diagonal: `mu(p)*mu(q) >= 0`
- 2 sum constraints:
  - `mu(123)*mu(1|2|3) + mu(12|3)*mu(1|23) >= 0`
  - `mu(123)*mu(1|2|3) + mu(12|3)*mu(13|2) >= 0`

### m=4: 17 inequalities (the paper's full result)

Includes the paper's key inequalities:

**Inequality (12) / Aas conjecture:**
```
mu(123)*mu(1|2|3) - mu(12|3)*mu(13|2) - mu(12|3)*mu(1|23) - mu(13|2)*mu(1|23) >= 0
```

**Inequality (11):**
```
mu(123)*mu(12|3) + mu(123)*mu(13|2) + mu(123)*mu(1|23) - mu(123)*mu(1|2|3)
+ 2*mu(12|3)*mu(13|2) + mu(12|3)*mu(1|23) + mu(13|2)*mu(1|23)
+ mu(1|23)*mu(1|2|3) >= 0
```

## Supporting Functions

The `lp_pipeline.jl` module provides functions used by the pipeline and also useful
independently for certificate construction and verification:

### Tuple handling

- `load_feasible_tuples()` -- load the bundled feasible tuples and normalize them on load to the paper tree order
- `load_feasible_tuples_raw(path, key)` -- read the raw legacy archive without normalization
- `archive_tuple_to_paper_pairs(tuple, n_obs, m)` -- legacy compatibility helper for raw archive tuples
- `phi_index(k, p, pbar, n_obs) -> Int` -- column index: `(k-1)*bell^2 + p*bell + pbar + 1`
- `inverse_phi_index(index, n_obs) -> (tree, p, pbar)` -- inverse mapping

### Constraint matrix

- `build_constraint_matrix(tuples, n_obs, m) -> SparseMatrixCSC` -- shape `(|F|, m*bell^2)`

### Certificate operations

- `verify_certificate(phi_tables, tuples; n_obs, m)` -- check nonnegativity over F
- `extract_quadratic_polynomial(phi_tables, n_obs)` -- aggregate phi to symmetric form
- `canonicalize_integer_ray(values)` -- rationalize to coprime integers
- `polynomial_signature(phi_tables, n_obs)` -- canonical signature for deduplication

### LP search (alternative to full enumeration)

- `find_inequality(M_F, objective, normalization; n_obs, m)` -- single LP for one inequality
- `systematic_inequality_search(M_F, n_obs, m; ...)` -- sweep over anchor/direction pairs

### Paper certificates

- `appendixA_certificate_11()`, `appendixA_certificate_12()` -- explicit integer certificates
- `inequality7_proof_potentials()` -- 4 potential tables from Proposition 10.1

## Solver Configuration

| Solver | Speed | License | Recommended for |
|--------|-------|---------|-----------------|
| HiGHS | Good | Free | n=3, all m |
| Gurobi | Best | Commercial | n=4, large instances |

The pipeline does **not** silently fall back from Gurobi to HiGHS when Gurobi is requested explicitly; Gurobi failures are surfaced directly.

### Typical performance (n=3)

| m | |F| | Rays | Time (HiGHS) | Iterations |
|---|-----|------|-------------|------------|
| 1 | 25  | 15   | < 1s        | 1          |
| 2 | 139 | 15   | < 1s        | 1          |
| 4 | 1265| 17   | ~4s         | 2          |

## Current Limitations

1. **n=3 only:** Hard-gated to the bundled paper-family archive. n=4 support requires
   extending the tuple loader/projector path to handle n=4 archives.
2. **m <= 4:** Limited by the archived m=4 dataset.
3. **No live enumeration:** Tuples must come from the pre-serialized `.jls` file.
   To use custom tuples, call the internal functions directly:
   ```julia
   M_F = build_constraint_matrix(your_tuples, n_obs, m)
   ext = PercolationOracle._extend_with_symmetric_variables(M_F, n_obs, m)
   result = projected_cone_oracle(ext.A, ext.projection_indices; config=config, solver=solver)
   ```
4. **No checkpointing passthrough:** The `OracleConfig.checkpoint_file` parameter
   is not exposed through `enumerate_all_inequalities`.
5. **Fixed projection:** Always uses symmetric (S_{p,q}) projection. No option for
   full phi-space projection.

## Testing

```bash
cd code/julia
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite includes two inequality enumeration testsets:
- **m=2:** Verifies 15 rays with exact integer signatures
- **m=4:** Verifies 17 rays and that Inequality (12) is among them
