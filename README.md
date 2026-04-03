# Percolation Inequality Oracle

Computational machinery for discovering and certifying universal polynomial inequalities
satisfied by connectivity partition probabilities in Bernoulli bond percolation.

This repository accompanies the paper:

> I. Gladkov, A. Zimin. *Bond-site percolation inequalities via decision-tree splicing*. 2026.

## The Problem

Consider Bernoulli bond percolation on a finite graph $G$ with $n$ distinguished
terminals $1, \dots, n$.  The random partition of $[n]$ into connectivity classes
defines a probability measure $\mu$ on the lattice of set partitions $\mathcal{J}_n$.
A *universal inequality* is a polynomial inequality in the $\mu(p)$ that holds for
every graph and every edge-retention parameter.

For $n = 3$, the partition lattice has $\lvert\mathcal{J}_3\rvert = 5$ elements:

$$\lbrace 123\rbrace,\quad \lbrace 1\mid 2\mid 3\rbrace,\quad \lbrace 1\mid 23\rbrace,\quad \lbrace 12\mid 3\rbrace,\quad \lbrace 13\mid 2\rbrace.$$

The question: what are all polynomial relations among $\mu(123), \mu(12\mid 3), \mu(13\mid 2), \mu(1\mid 23), \mu(1\mid 2\mid 3)$?

## The Algorithm

The answer is computed in two stages.

### Stage D1: Decision-Tree Feasibility Oracle ([PercolationOracle.jl](PercolationOracle/))

Fix a family of $m$ *good decision trees* $T_1, \dots, T_m$ operating on the
condensation graph $\bar{V} = [n] \cup \lbrace\star\rbrace$.  Each tree $T_k$ induces a
2-coloring $\mathrm{col}_k : \bar{V} \times \bar{V} \to \lbrace 1, 2\rbrace$ that partitions
edges into two splicing regions.

The oracle decides which partition $m$-tuples $\mathbf{p} = (p_1, \dots, p_m) \in \mathcal{J}_n^m$
are *realizable* via decision-tree splicing.  It does so by a BFS reachability
algorithm on the universal state space

$$\mathcal{S} = \bar{V} \times \prod_{k=1}^{m} \lbrace 0, 1, \dots, r_k\rbrace$$

with transition relations encoding $G_1$-steps, $G_2$-steps, and a blocking
condition.  A tuple is *accepted* if $G_1$-compatibility holds and every block
of every partition passes the reachability test.

For the paper's four-tree family $(T_0, T_1, T_2, T_3)$ at $n = 3$, the doubled
instance $(T_0, \dots, T_3, \bar{T}_0, \dots, \bar{T}_3)$ yields a feasible set of

$$\lvert F\rvert = 1265 \text{ partition tuples.}$$

### Stage D2: Projected Cone Oracle ([ProjectedConeOracle.jl](ProjectedConeOracle/))

The feasible set defines a polyhedral cone via the feasible-potentials framework:

$$C = \bigl\lbrace \varphi \in \mathbb{R}^{m \cdot \lvert\mathcal{J}_n\rvert^2} : \sum_{k} \varphi_k(p_k, \bar{p}_k) \ge 0 \forall \mathbf{p} \in F \bigr\rbrace.$$

Each feasible $\varphi$ gives a universal inequality for the partition probabilities.
To express these as polynomial inequalities in $\mu(p)$, we project $C$ to the
symmetric coordinates

$$S_{p,q} = \mu(p) \cdot \mu(q),$$

yielding a $\tbinom{\lvert\mathcal{J}_n\rvert+1}{2}$-dimensional projected cone $D = \pi_J(C)$.

ProjectedConeOracle computes $D$ via an oracle-driven double-description algorithm:
iterative LP separation discovers extreme rays while Normaliz enumerates facets.
The extreme rays of $D$ are exactly the universal inequalities.

## Main Results

For $n = 3$ and $m = 4$ decision trees, the algorithm produces **17 universal inequalities**.
The 15 inequalities at $m \le 2$ are trivially nonnegative monomials and cross-terms.
At $m = 4$, two nontrivial inequalities emerge:

**Inequality (12) / Aas Conjecture** ([Gladkov 2023, Corollary 4.2](https://arxiv.org/abs/2310.18263)):

$$\mu(123)\mu(1\mid 2\mid 3) - \mu(12\mid 3)\mu(13\mid 2) - \mu(12\mid 3)\mu(1\mid 23) - \mu(13\mid 2)\mu(1\mid 23) \ge 0$$

**Inequality (11)**:

$$\mu(123)\mu(12\mid 3) + \mu(123)\mu(13\mid 2) + \mu(123)\mu(1\mid 23) - \mu(123)\mu(1\mid 2\mid 3) + 2\mu(12\mid 3)\mu(13\mid 2) + \mu(12\mid 3)\mu(1\mid 23) + \mu(13\mid 2)\mu(1\mid 23) + \mu(1\mid 23)\mu(1\mid 2\mid 3) \ge 0$$

**Inequality (7)** (Proposition 10.1):

$$\mu(1\mid 2 \cap 1\mid 3)\mu(12 \cup 13) \le \mu(12\mid 3) + \mu(13\mid 2) + \mu(1\mid 23)$$

All inequalities come with explicit integer feasible potentials (certificates) that
can be verified computationally.

## Quick Start

```julia
using Pkg
Pkg.develop(path="ProjectedConeOracle")
Pkg.develop(path="PercolationOracle")

using PercolationOracle

# Enumerate all universal inequalities for n=3, m=4
result = enumerate_all_inequalities(3, 4)

@assert result.projection_result.converged
@assert length(result.rays) == 17

# Print the inequalities
for ineq in result.formatted_inequalities
    println(ineq)
end
```

Output:
```
mu(123)*mu(1|2|3) - mu(12|3)*mu(13|2) - mu(12|3)*mu(1|23) - mu(13|2)*mu(1|23) >= 0
mu(123)*mu(12|3) + mu(123)*mu(13|2) + mu(123)*mu(1|23) - mu(123)*mu(1|2|3) + 2*mu(12|3)*mu(13|2) + mu(12|3)*mu(1|23) + mu(13|2)*mu(1|23) + mu(1|23)*mu(1|2|3) >= 0
mu(123)^2 >= 0
mu(12|3)^2 >= 0
...
```

### Verify an explicit certificate

```julia
# Verify the Appendix A certificate for Inequality (11)
F = paper_family_feasible_tuples()
cert = appendixA_certificate_11()
v = verify_certificate(cert, F; n_obs=3, m=4)
@assert v.feasible && v.minimum == 0
```

## Installation

Requires **Julia 1.11+**.

```julia
using Pkg

# Clone the repository, then from the repo root:
Pkg.develop(path="ProjectedConeOracle")  # dependency, install first
Pkg.develop(path="PercolationOracle")
```

The default LP solver is [HiGHS](https://github.com/jump-dev/HiGHS.jl) (free, installed
automatically).  For optional Gurobi support:

```julia
Pkg.add("Gurobi")  # requires a Gurobi license
```

## Packages

| Package | Role | Description |
|---------|------|-------------|
| [PercolationOracle](PercolationOracle/) | D1 | Decision-tree feasibility oracle: enumerates the feasible set $F$, builds the constraint matrix $M_F$, produces and verifies inequality certificates |
| [ProjectedConeOracle](ProjectedConeOracle/) | D2 | General-purpose projected cone computation: LP separation + Normaliz facet enumeration, with subspace reduction, checkpointing, and multiple solver backends |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`reproduce_paper_results.ipynb`](notebooks/reproduce_paper_results.ipynb) | End-to-end reproduction of the paper's main computational results |
| [`inequality_enumeration.ipynb`](notebooks/inequality_enumeration.ipynb) | Interactive walkthrough of the D1 &rarr; D2 pipeline with intermediate outputs |

## Performance

All timings on a single machine with HiGHS (free LP solver).

| m | \|F\| | Rays | D2 Time | D2 Iterations |
|---|-------|------|---------|---------------|
| 1 | 25    | 15   | < 1 s   | 1             |
| 2 | 139   | 15   | < 1 s   | 1             |
| 3 | 570   | 15   | ~ 1 s   | 1             |
| 4 | 1,265 | 17   | ~ 4 s   | 2             |

The D1 enumeration (computing $F$ from scratch via BFS) takes approximately 45 seconds
for $m = 4$ with 4 threads.  Pre-computed feasible tuples are included in
[`PercolationOracle/data/`](PercolationOracle/data/).

## Testing

```bash
# Test ProjectedConeOracle
cd ProjectedConeOracle
julia --project -e 'using Pkg; Pkg.test()'

# Test PercolationOracle
cd ../PercolationOracle
julia --project -e 'using Pkg; Pkg.test()'
```

## Repository Structure

```
.
├── PercolationOracle/          D1: decision-tree feasibility oracle
│   ├── src/                    Core algorithm (11 modules)
│   ├── test/                   Test suite (11 files)
│   ├── data/                   Pre-computed feasible tuple archives
│   ├── docs/                   API documentation
│   └── benchmarks/             Performance benchmarks
├── ProjectedConeOracle/        D2: projected polyhedral cone oracle
│   ├── src/                    Core algorithm (12 modules)
│   ├── test/                   Test suite (11 files)
│   └── ext/                    Solver extensions (Gurobi, GLPK)
├── notebooks/                  Jupyter notebooks reproducing results
└── legacy/                     Original baseline code (2024)
```

## Legacy Code

The [`legacy/`](legacy/) directory contains the original 2024 implementation:
- `decision_trees.jl` --- baseline decision-tree algorithm module
- `decision_tree_algorithm.ipynb` --- original notebook recovering inequalities (5) and (6) via Gurobi integer programming

These are superseded by the packages above but preserved for reference.

## Citation

```bibtex
@article{gladkovzimin2026bondsite,
  title   = {Bond-site percolation inequalities via decision-tree splicing},
  author  = {Gladkov, Ivan and Zimin, Aleksandr},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
