# Odd QSP Approximation with an Exact Bound Certificate

7/17/2026

## Problem

Let $d=2m-1$ be odd and represent the QSP polynomial by its partial Chebyshev coefficients,

$$
F(x)=\sum_{j=0}^{m-1}c_jT_{2j+1}(x).
$$

For a finite grid $\mathcal{K}=\{x_i\}$, let $I$ index the points in the approximation region $\mathcal{I}\subseteq[0,1]$. We define $A_{ij}=T_{2j+1}(x_i)$, set $f=Ac$, and impose the polynomial bound on the full interval:

$$
\begin{aligned}
\underset{c}{\operatorname{minimize}}\quad & \lVert f[I]-g[I]\rVert_\infty \\
\text{subject to}\quad & f=Ac, \\
& -1\leq F(x)\leq1 \quad \text{for every }x\in[-1,1].
\end{aligned}
$$

The grid discretizes only the approximation error, not the bound constraint.

## Matrix Certificate

Foucart and Powers characterize nonnegative Chebyshev polynomials by positive semidefinite Gram matrices [1](#Foucart_basc). On the full interval $[-1,1]$, Theorem 3 discards its interval-localizing matrix. If

$$
P(x)=\sum_{k=0}^{d}p_kT_k(x),
$$

then $P(x)\geq0$ on $[-1,1]$ if and only if a Hermitian matrix $Q\succeq0$ satisfies

$$
p_0=\sum_{i=0}^{d}Q_{ii}, \qquad \frac{p_k}{2}=\sum_{i-j=k}Q_{ij}, \quad 1\leq k\leq d.
$$

We apply the theorem to $P(x)=1-F(x)$. If $P(x)\geq0$ for every $x$, oddness gives

$$
P(-x)=1-F(-x)=1+F(x)\geq0.
$$

Thus $P\geq0$ is equivalent to both $F\leq1$ and $F\geq-1$. Its Chebyshev coefficients are

$$
p_0=1, \qquad p_{2j+1}=-c_j, \qquad p_{2j}=0 \quad \text{for }j\geq1.
$$

The theorem initially permits a complex Hermitian Gram matrix. We write $Q=X+\mathrm{i}Y$. For every real vector $v$, positive semidefiniteness gives $v^{\mathsf{T}}Xv=v^*Qv\geq0$, so $X=\operatorname{Re}(Q)$ is real symmetric positive semidefinite. Since every required diagonal sum is real, $X$ satisfies the same coefficient identities. We may therefore use a real matrix without loss.

We define

$$
L_k(Q)=\sum_{i=0}^{d-k}Q_{i,i+k}.
$$

The exact bound $|F(x)|\leq1$ is equivalent to the existence of $Q\in\mathbb{S}_+^{d+1}$ satisfying

$$
\begin{aligned}
L_0(Q)&=1, \\
L_k(Q)&=-\frac{c_{(k-1)/2}}{2} && \text{for odd }k, \\
L_k(Q)&=0 && \text{for positive even }k.
\end{aligned}
$$

Oddness removes the second nonnegativity certificate that would otherwise enforce $1+F\geq0$.

## Semidefinite Program and Algorithm

We combine the unchanged grid objective with the one-matrix certificate:

$$
\begin{aligned}
\underset{c,f,Q}{\operatorname{minimize}}\quad & \lVert f[I]-g[I]\rVert_\infty \\
\text{subject to}\quad & f=Ac, \\
& Q\succeq0, \\
& L_0(Q)=1, \\
& L_k(Q)=-\frac{c_{(k-1)/2}}{2} && \text{for odd }k, \\
& L_k(Q)=0 && \text{for positive even }k.
\end{aligned}
$$

The implementation performs the following operations.

1. We validate that $d$ is positive and odd and that the fit intervals are ordered subsets of $[0,1]$.
2. We construct the notebook's nonnegative Chebyshev grid, insert the fit-interval endpoints, and form $A_{ij}=T_{2j+1}(x_i)$.
3. We create $c$, $f$, and a real symmetric $Q$, retain `f == A @ c` and `cp.norm(f[I] - g[I], inf)`, and add only the Gram constraints for the bound.
4. We solve the SDP with SCS and independently check the diagonal sums, the minimum Gram eigenvalue, $|F|$ on $200001$ Chebyshev-distributed points, and $|F|$ at all real critical points obtained from $F'$.
5. If roundoff produces a negative Gram eigenvalue or a maximum too close to $1$, we apply the structure-preserving restoration below.
6. We accept a numerical result only if the diagonal residual is at most $10^{-5}$, the restored Gram matrix has nonnegative minimum eigenvalue, and the computed global maximum satisfies $\max_{[-1,1]}|F(x)|\leq1$ without an additive tolerance.

The core CVXPY construction in [constrainapprox.py](constrainapprox.py) is

```python
coefficients = cp.Variable((degree + 1) // 2, name="c")
values = cp.Variable(grid.size, name="f")
gram = cp.Variable((degree + 1, degree + 1), symmetric=True, name="Q")

constraints = [values == design @ coefficients, gram >> 0]
constraints.append(cp.trace(gram) == bound)
for k in range(1, degree + 1):
    diagonal_sum = cp.sum(cp.diag(gram, k=k))
    if k % 2:
        constraints.append(diagonal_sum == -0.5 * coefficients[(k - 1) // 2])
    else:
        constraints.append(diagonal_sum == 0.0)

objective = cp.Minimize(
    cp.norm(values[fit_indices] - target_values[fit_indices], "inf")
)
problem = cp.Problem(objective, constraints)
```

### Strict-feasibility restoration

A spectral projection of $Q$ onto the positive semidefinite cone generally changes its diagonal sums and breaks the identities linking $Q$ to $c$. Instead, let $b$ denote the requested bound and use the known certificate $Q_0=bI_{d+1}/(d+1)$ for the constant polynomial $b$. For a scale $0\leq\alpha\leq1$, we set

$$
c_{\alpha}=\alpha c, \qquad Q_{\alpha}=\alpha Q+(1-\alpha)\frac{b}{d+1}I_{d+1}.
$$

In exact arithmetic, $Q_{\alpha}$ preserves the trace and diagonal-sum identities for $b-\alpha F$, while the positive identity component moves a slightly indefinite matrix into the positive semidefinite cone. The code chooses the largest $\alpha$ that places the minimum eigenvalue at least $10^{-8}$ and the computed global maximum below $b$ by at least $10^{-8}$. It then recomputes the objective and all certificate diagnostics from the returned coefficients.

The complete script uses CVXPY and runs in the requested environment as follows:

```bash
python constrainapprox.py --degree 101 --npts 500 --solver SCS
```

The public function can be called directly:

```python
from constrainapprox import solve_odd_bounded_approx

a = 0.2
delta = 0.01
target = lambda x: 0.9 * x / a
coefficients, diagnostics = solve_odd_bounded_approx(
    target,
    degree=101,
    fit_intervals=[0.0, a - delta],
    npts=500,
    bound=1.0,
    solver="SCS",
)
```



## Numerical Validation

For a small-degree check, we used $d=5$, $g(x)=x/2$, and $\mathcal{I}=[0,0.5]$. SCS recovered the exact linear target to a grid error of $3.07\times10^{-8}$. The diagonal-sum residual was $2.99\times10^{-11}$, the minimum Gram eigenvalue was $7.73\times10^{-2}$, and the global maximum of $|F|$ was $0.499994$. A synthetic degree-$1$ roundoff violation also verified that the restoration produces a positive semidefinite Gram matrix and a maximum below $1$. 

For uniform singular value amplification, the notebook uses a linear target and a degree-$101$ odd polynomial [2](#uniform_singular_value_amplification). We used

$$
a=0.2, \qquad \delta=0.01, \qquad g(x)=0.9\frac{x}{a}, \qquad \mathcal{I}=[0,a-\delta]=[0,0.19].
$$

The reference grid has $502$ nonnegative points after endpoint insertion, of which $63$ lie in $\mathcal{I}$. SCS solved the SDP with `eps=1e-6` and `max_iters=200000`. The comparison problem retained the original sampled inequalities with $\eta=0$ and used CLARABEL with feasibility and gap tolerances $10^{-7}$ because SCS reached its iteration limit on that LP.



| Method | Role | Continuous bound satisfied | Grid error | Grid maximum of $|F|$ | Computed global maximum of $|F|$ | Diagonal residual | Minimum Gram eigenvalue | Iterations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exact SDP bound | Accepted solution | Yes | $1.8198\times10^{-6}$ | $0.999791$ | $0.999872$ | $2.2829\times10^{-10}$ | $1.2247\times10^{-6}$ | $1675$ |
| Sampled bound | Diagnostic baseline only | No | $3.7780\times10^{-7}$ | $0.999977$ | $1.000391$ | N/A | N/A | $17$ |



## References

<a id="Foucart_basc">[1]</a> Simon Foucart and Vladlena Powers, "Basc: Constrained Approximation by Semidefinite Programming," 2016.

<a id="uniform_singular_value_amplification">[2]</a> pyqsppack, "Uniform Singular Value Amplification." [Local notebook](/home/linlin/Projects/pyqsppack/docs/source/examples/uniform_singular_value_amplification.ipynb).
