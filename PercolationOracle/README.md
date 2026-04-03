# PercolationOracle.jl

**Decision-tree feasibility oracle for universal bond-percolation inequalities.**

Part of the [Kroneckera/bunkbed](https://github.com/Kroneckera/bunkbed) repository.
Companion package: [ProjectedConeOracle.jl](../ProjectedConeOracle/).

---

## Overview

PercolationOracle.jl implements the decision-tree feasibility oracle (Stage D1) from the two-stage algorithm of Gladkov--Zimin (2026) for discovering universal inequalities on bond-percolation connectivity probabilities.

**The mathematical setup.** Fix $n$ terminal vertices and consider Bernoulli bond percolation on an arbitrary finite graph $G$ containing those terminals. The random connectivity partition $\pi \in \mathcal{J}_n$ records which terminals belong to the same connected component. Its law $\mu = \mathrm{Law}(\pi)$ is a probability measure on the lattice of set partitions $\mathcal{J}_n$, and the set of all laws achievable by some $(G, p)$ forms a convex body $\mathcal{M}_n \subset \Delta(\mathcal{J}_n)$.

**What the oracle decides.** Given $m$ decision trees $T_1, \dots, T_m$ operating on a condensation graph $\bar{V} = [n] \cup \lbrace *\rbrace$, the oracle determines which $m$-tuples of partition pairs $((\pi_1, \bar{\pi}_1), \dots, (\pi_m, \bar{\pi}_m)) \in (\mathcal{J}_n^2)^m$ are *realizable* --- that is, achievable by some decision-tree splicing construction. The set of all such realizable tuples is the *feasible set* $\mathcal{F}$.

**Role in the two-stage pipeline.** The feasible set $\mathcal{F}$ produced by this oracle (D1) feeds into [ProjectedConeOracle.jl](../ProjectedConeOracle/) (D2), which computes the projected cone of valid potential-function inequalities and extracts the extreme rays --- the universal percolation inequalities.

---

## Mathematical Formulation

This section describes the core algorithm from Gladkov--Zimin (2026). All notation follows the paper.

### Condensation graph and decision trees

Fix $n$ observables (terminal vertices). The *condensation graph* is the complete graph on $\bar{V} = \lbrace 1, \dots, n, *\rbrace$, where $*$ represents the "rest of the world" vertex. A partition $\pi \in \mathcal{J}_n$ of $[n]$ determines which edges of $\bar{V}$ are present in the condensed graph: vertices $u, v \in [n]$ are connected if and only if they belong to the same block of $\pi$.

A *decision tree* $T$ on $\bar{V}$ is a sequence of steps, each specifying a set of seed vertices and a graph label $g \in \lbrace 1, 2\rbrace$. The tree is applied to a condensation via BFS: starting from the seeds, it explores edges of the condensation and assigns a 2-coloring $\mathrm{col}_T : \bar{V} \times \bar{V} \to \lbrace 1, 2\rbrace$ to every pair of vertices.

### Oracle state space and transitions

Fix $m$ decision trees $T_1, \dots, T_m$ on $\bar{V}$, each inducing a 2-coloring $\mathrm{col}_k$. The oracle operates on the state space

$$S = \bar{V} \times \prod_{k=1}^{m} \lbrace 0, 1, \dots, r_k\rbrace$$

where $r_k$ is one more than the number of blocks of partition $\pi_k$. A state $(v, \psi_1, \dots, \psi_m)$ encodes the current vertex $v$ together with a "tracking value" $\psi_k$ for each tree $k$: the value $\psi_k = 0$ means "uncommitted", while $\psi_k = b > 0$ means "committed to block $b$ of $\pi_k$".

Three types of transitions govern the BFS exploration:

1. **Star-loop** ($v = * \to *$): The walk stays at $*$ and resets tracking values according to the diagonal coloring $\mathrm{col}_k(*, *)$.

2. **G1-step** ($\mathrm{col}_{\mathrm{active}}(v, w) = 1$): A step along graph 1 (the condensation). This is allowed only if the edge $(v, w)$ exists in the condensation. The tracking values for passive trees are determined by the transition table: the mask $\sigma_k(w)$ reads off the block assignment of $w$ in $\pi_k$ when $\mathrm{col}_k(w, w) = 1$, and the new $\psi_k$ value is set to $\sigma_k(w)$.

3. **G2-step** ($\mathrm{col}_{\mathrm{active}}(v, w) = 2$): A step along graph 2 (the complement). The tracking value $\psi_k$ either stays as is or gets refined to $\sigma_k(w)$, unless there is a conflict ($\psi_k \neq 0$ and $\sigma_k(w) \neq 0$ and $\psi_k \neq \sigma_k(w)$), which *blocks* the transition.

### Acceptance condition

A partition $m$-tuple $(\pi_1, \dots, \pi_m)$ is *accepted* (oracle-feasible) if two conditions hold:

1. **G1-compatibility**: For each tree $k$, no edge $(u, v)$ with $\mathrm{col}_k(u, v) = 1$ connects two vertices that lie in *different* blocks of $\pi_k$ while the condensation edge $(u, v)$ is present.

2. **BFS reachability**: For each tree $k$ and each block $B$ of $\pi_k$, starting BFS from any vertex in $B$, every other vertex of $B$ is reachable via the transition rules while maintaining compatibility with the tracking values of all $m$ trees simultaneously.

### Doubled instance and feasible-potentials LP

Each tree $T_k$ and its complement $\bar{T}_k$ (obtained by swapping graph labels $1 \leftrightarrow 2$) together produce a partition *pair* $(\pi_k, \bar{\pi}_k)$. The oracle runs on $2m$ colorings --- the $m$ original trees and their $m$ complements --- yielding partition tuples of length $2m$.

The feasible set $\mathcal{F} \subset (\mathcal{J}_n^2)^m$ indexes the rows of a constraint matrix $M\_\mathcal{F}$, and the universal inequalities are extreme rays of the projected cone

$$\lbrace x : M\_\mathcal{F} \, x \geq 0\rbrace$$

restricted to the symmetric polynomial coordinates

$$S\_{p,q} = \sum\_k \bigl[\varphi\_k(p, q) + \varphi\_k(q, p)\bigr].$$

### The paper four-tree family

The paper's main computation uses $n = 3$ terminals and $m = 4$ decision trees $T_0, T_1, T_2, T_3$ at condensation level, together with their complements $\bar{T}_0, \bar{T}_1, \bar{T}_2, \bar{T}_3$ (8 colorings total). The trees are:

| Tree | Steps | Description |
|------|-------|-------------|
| $T_0$ | $(\lbrace 1,2,3,4\rbrace , g=1)$ | All seeds, graph 1 |
| $T_1$ | $(\lbrace 3\rbrace , 1), (\lbrace 1\rbrace , 2), (\lbrace 2\rbrace , 1), (\lbrace 4\rbrace , 2)$ | Alternating seeds |
| $T_2$ | $(\lbrace 2\rbrace , 1), (\lbrace 1\rbrace , 2), (\lbrace 3\rbrace , 1), (\lbrace 4\rbrace , 2)$ | Alternating seeds |
| $T_3$ | $(\lbrace 1\rbrace , 1), (\lbrace 2,3\rbrace , 2), (\lbrace 4\rbrace , 1)$ | Mixed seeds |

Here vertex 4 represents $*$ in $\bar{V} = \lbrace 1, 2, 3, *\rbrace$.

---

## Main Results

For $n = 3$ observables and $m = 4$ decision trees:

- The oracle enumerates $\lvert\mathcal{F}\rvert = 1265$ feasible partition tuples in $(\mathcal{J}_3^2)^4$.
- Feeding $\mathcal{F}$ into the projected cone oracle (D2) yields **17 extreme rays** = 17 universal inequalities on $\mu(\pi)^2$-monomials.
- These include:
  - **Inequality (12)** (the Aas conjecture): $\mu(123)\mu(1{\mid}2{\mid}3) - \mu(12{\mid}3)\mu(13{\mid}2) - \mu(12{\mid}3)\mu(1{\mid}23) - \mu(13{\mid}2)\mu(1{\mid}23) \geq 0$
  - **Inequality (11)**: $\mu(123)\mu(12{\mid}3) + \mu(123)\mu(13{\mid}2) + \mu(123)\mu(1{\mid}23) - \mu(123)\mu(1{\mid}2{\mid}3) + 2\mu(12{\mid}3)\mu(13{\mid}2) + \mu(12{\mid}3)\mu(1{\mid}23) + \mu(13{\mid}2)\mu(1{\mid}23) + \mu(1{\mid}23)\mu(1{\mid}2{\mid}3) \geq 0$

Scaling by prefix length $m$:

| $m$ | $\lvert\mathcal{F}\rvert$ | Rays | D1 time (single-threaded) |
|-----|-----------------|------|---------------------------|
| 1   | 25              | 15   | 0.02 ms                   |
| 2   | 139             | 15   | 0.21 ms                   |
| 3   | 570             | --   | 2.7 ms                    |
| 4   | 1,265           | 17   | 14.7 ms                   |

---

## Quick Start

```julia
using PercolationOracle
using ProjectedConeOracle

# Run the full D1 -> D2 pipeline: 3 terminals, 4 decision trees
result = enumerate_all_inequalities(3, 4)

# Inspect the output
println("Converged: ", result.projection_result.converged)
println("Number of inequalities: ", length(result.rays))
for ineq in result.formatted_inequalities
    println(ineq)
end
```

To run the D1 oracle alone (feasible-tuple enumeration without D2):

```julia
using PercolationOracle

# Enumerate the 1,265 feasible tuples via the paper family regression
reg = enumerate_paper_family_regression()
println("Feasible tuples: ", reg.count)     # 1265
println("SHA-1 hash:      ", reg.hash)
```

To verify an explicit certificate against the feasible set:

```julia
using PercolationOracle

tuples = load_feasible_tuples()
cert = appendixA_certificate_12()   # Inequality (12) / Aas conjecture
result = verify_certificate(cert, tuples; n_obs=3, m=4)
println("Certificate valid: ", result.feasible)  # true
println("Minimum value:     ", result.minimum)    # 0
```

---

## Installation

PercolationOracle.jl depends on [ProjectedConeOracle.jl](../ProjectedConeOracle/), which lives in the same repository. Install both as development packages:

```julia
using Pkg
Pkg.develop(path="/path/to/bunkbed/ProjectedConeOracle")
Pkg.develop(path="/path/to/bunkbed/PercolationOracle")
```

**Requirements:**

- Julia >= 1.11
- Dependencies (automatically resolved): Combinatorics, DataStructures, HiGHS, JuMP, ProgressMeter, ProjectedConeOracle, SHA

---

## API Reference

### `enumerate_all_inequalities`

Run the full D1-to-D2 pipeline. Loads the included legacy feasible-tuple archive, normalizes it on load to the paper tree order \((T_0,T_1,T_2,T_3)\), builds the constraint matrix, extends with symmetric polynomial variables, and passes to the projected cone oracle.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_obs | Int | required | Number of terminals (currently only 3) |
| m | Int | required | Number of decision trees (1 to 4) |
| solver | AbstractSolverAdapter | HiGHSAdapter() | LP solver backend |
| verbose | Bool | true | Print iteration-level progress |
| max_iterations | Int | 5000 | Oracle iteration cap |

Returns an InequalityEnumerationResult:

| Field | Type | Description |
|-------|------|-------------|
| rays | Vector{Vector{Float64}} | Extreme rays = universal inequalities in symmetric coordinates |
| facet_normals | Vector{Vector{Float64}} | Facet normals of the projected cone |
| labels | Vector{String} | Monomial labels, e.g. "mu(123)^2", "mu(12\|3)*mu(1\|23)" |
| formatted_inequalities | Vector{String} | Human-readable ">= 0" strings |
| projection_result | ProjectedConeResult | Raw D2 output (rays, facets, iterations, converged) |
| timing | Float64 | Wall-clock seconds for the D2 stage |

### `enumerate_paper_family_regression`

Enumerate all 1265 feasible partition tuples for the paper four-tree family (n=3, m=4, 8 colorings). Returns `(tuples, counts, hash, count)` with SHA-1 digest for regression testing. A multi-threaded variant `enumerate_paper_family_regression_threaded(; n_threads)` distributes work across Julia threads.

### `paper_family_decision_trees`

Return the 8 decision trees (4 trees + 4 complements) used in the paper computation.

### `verify_certificate`

Check that a potential-function certificate satisfies nonnegativity over the feasible set. Takes `(phi_tables, tuples; n_obs, m)`. Returns `(feasible, minimum, negatives, witness_indices, witness_tuples)`.

### `appendixA_certificate_11` / `appendixA_certificate_12`

Explicit integer potential tables from Appendix A of the paper. Certificate 11 corresponds to Inequality (11); certificate 12 corresponds to the Aas conjecture.

### `inequality7_proof_potentials`

Potential tables from the proof of Inequality (7) (Proposition 10.1). These use different trees than the paper family and are not feasible on the paper's feasible set, but serve as a proof-of-concept.

### `build_constraint_matrix`

Build the sparse constraint matrix from feasible tuples. Takes `(tuples, n_obs, m)`. Shape: (number of tuples) x (m * Bell(n)^2). Each row has exactly m nonzero entries.

### `find_inequality` / `systematic_inequality_search`

LP-based inequality discovery. `find_inequality` solves a single LP to find one inequality. `systematic_inequality_search` sweeps over anchor/direction pairs to discover distinct inequalities, deduplicating by polynomial signature.

### `load_feasible_tuples`

Load feasible tuples from the included `.jls` archive and normalize them on load to the paper tree order \((T_0,T_1,T_2,T_3)\). For raw archive access, use `load_feasible_tuples_raw`.

### `canonicalize_integer_ray`

Rationalize floating-point values to coprime integers. Used to produce exact certificates from LP solutions.

---

## Data Files

Archived feasible-tuple datasets live in `data/`:

| File | Format | Description |
|------|--------|-------------|
| `valid_partition_tuples_nobs3_notebook.jls` | Julia `Serialization` | Binary archive of 1,265 feasible 8-tuples (Dict with key `"tuples_ids_prefix"`) |
| `valid_partition_tuples_nobs3_notebook.tsv` | Tab-separated | 1-based partition IDs (8 columns) + human-readable representation |
| `valid_partition_tuples_nobs3_notebook.txt` | Pipe-delimited | Human-readable partition labels, e.g. `123 \|\| 12\|3 \|\| ...` |
| `valid_partition_tuples_nobs2_restricted.*` | Same formats | Restricted feasible tuples for $n = 2$ |

The `.jls`, `.tsv`, and `.txt` files are legacy raw archives. Their tree entries are stored in the historical order $(\pi_0, \bar{\pi}_0, \pi_2, \bar{\pi}_2, \pi_1, \bar{\pi}_1, \pi_3, \bar{\pi}_3)$. The public loader `load_feasible_tuples()` normalizes these tuples on load to the paper order $(T_0, T_1, T_2, T_3)$; use `load_feasible_tuples_raw()` only when you explicitly need the raw archive format.

---

## Performance

### D1: Feasible-tuple enumeration

Benchmarks on 4 Julia threads (from `benchmarks/bench_by_m.log`):

| $m$ | Free colorings | $\lvert\mathcal{F}\rvert$ | Single-threaded | 4-threaded | Speedup |
|-----|----------------|-----------------|-----------------|------------|---------|
| 1   | 1              | 25              | 0.018 ms        | --         | --      |
| 2   | 3              | 139             | 0.206 ms        | --         | --      |
| 3   | 5              | 570             | 2.74 ms         | --         | --      |
| 4   | 7              | 1,265           | 14.7 ms         | 12.0 ms    | 1.23x   |

### D1 + D2: Full pipeline

| $m$ | $\lvert\mathcal{F}\rvert$ | Rays | D2 time (HiGHS) |
|-----|-----------------|------|------------------|
| 1   | 25              | 15   | < 1 s            |
| 2   | 139             | 15   | < 1 s            |
| 4   | 1,265           | 17   | ~4 s             |

---

## Testing

```bash
cd PercolationOracle
julia --project -e 'using Pkg; Pkg.test()'
```

The test suite covers:

| Test file | Scope |
|-----------|-------|
| `test_partitions.jl` | Partition encoding/decoding, Bell numbers, canonical labels |
| `test_colorings.jl` | Decision-tree application, 2-coloring packing/unpacking |
| `test_packed_state.jl` | Bit-packed state representation, mixed-radix indexing |
| `test_transitions.jl` | Transition table construction, blocking conditions |
| `test_oracle.jl` | Core oracle acceptance checks (G1-compatibility, BFS reachability) |
| `test_regression.jl` | Full paper-family regression: 1,265 tuples with SHA-1 hash verification |
| `test_threaded.jl` | Multi-threaded enumeration correctness |
| `test_lp_pipeline.jl` | Constraint matrix, certificate verification, Appendix A certificates |
| `test_inequality_enumeration.jl` | End-to-end D1+D2 pipeline: $m = 2$ (15 rays), $m = 4$ (17 rays) |

---

## Architecture

```
src/
  PercolationOracle.jl      Module entry point
  partitions.jl              Partition encoding: J_n <-> PartitionID, Bell numbers,
                               block/assignment lookups via precomputed PartitionTable
  condensation.jl            Condensation graph: bit-packed adjacency (CondensationBits),
                               edge queries, partition-to-condensation mapping
  colorings.jl               Decision trees and 2-colorings: DecisionTreeStep, DecisionTree,
                               BFS-based tree application, bit-packed PackedColoring (UInt16)
  packed_state.jl            Bit-packed oracle state: component + m tracking values in a
                               single UInt32/UInt64, mixed-radix indexing for visited array
  transitions.jl             Precomputed transition table: next_psi lookup for all
                               (tree, partition, component, next_component, current_psi)
                               combinations; blocking sentinel BLOCKED_PSI
  oracle_packed.jl           Core oracle: G1-compatibility check + BFS reachability with
                               packed states, pre-allocated OracleWorkspace
  enumeration_packed.jl      Single-threaded backtracking enumeration over partition tuples
  enumeration_threaded.jl    Multi-threaded enumeration with per-thread workspaces and
                               deterministic result merging
  paper_family.jl            Paper four-tree family (T0..T3 + complements), regression
                               test driver with SHA-1 hash verification
  lp_pipeline.jl             LP operations: constraint matrix M_F, certificate verification,
                               find_inequality via HiGHS/JuMP, systematic search,
                               Appendix A certificates, integer canonicalization
  inequality_enumeration.jl  D1+D2 pipeline: symmetric extension, projected cone call,
                               result formatting as InequalityEnumerationResult
```

---

## Citation

If you use this package in your research, please cite:

```bibtex
@article{GladkovZimin2026,
  author  = {Gladkov, Nikita and Zimin, Aleksandr},
  title   = {Universal inequalities for bond-percolation connectivity probabilities},
  year    = {2026},
}
```

---

## License

See the [repository root](https://github.com/Kroneckera/bunkbed) for license information.
