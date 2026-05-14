# Plan: Piecewise Linear Dynamics of Regression Weight Trajectories

## Motivation

The regression weight vector $w_{t,u} \in \mathbb{R}^{d}$ encodes the *coding direction* in PCA feature
space that best predicts unit $u$'s response at time bin $t$.  Projecting the population weight
matrix $W_t \in \mathbb{R}^{N \times d}$ into its top-$k$ PCA subspace gives a low-dimensional
trajectory $z_t \in \mathbb{R}^k$ with participation ratio $\approx 4.3$.

A global LDS $z_{t+1} = A z_t$ fits one-step predictions well (R² = 0.99) but open-loop rollout
degrades rapidly (R² = 0.24), indicating the dynamics are **locally but not globally linear**.  Neural
responses have distinct temporal phases (pre-stimulus → onset → peak → sustained) — a natural
setting for **piecewise** linear dynamics.

---

## Phase 1 — Diagnose global LDS failure

**Goal:** find *where in time* the single-$A$ model breaks down.

- Fit global $A$ on full trajectory.
- Plot per-timestep residual $r_t = \|z_{t+1} - A z_t\|^2$ vs.\ $t$.
- Expected pattern: large residuals at onset (rapid acceleration) and offset (deceleration),
  small residuals during sustained period.

---

## Phase 2 — Fixed-segment piecewise LDS

**Goal:** quick validation that multiple regimes improve things.

Define $K = 3$ segments anchored to known response phases:

| Segment | Time range | Label |
|---------|-----------|-------|
| 1 | $t < 0$ ms | pre-stimulus |
| 2 | $0 \leq t < 150$ ms | onset |
| 3 | $t \geq 150$ ms | sustained |

For each segment $k$, fit:

$$A_k = \arg\min_{A} \sum_{t \in \mathcal{T}_k} \|z_{t+1} - A z_t\|^2$$

via ordinary least-squares.

**Metrics:**
- Per-segment one-step R²
- Open-loop rollout: reset $\hat{z}_{T_k} = z_{T_k}$ at each boundary, propagate with $A_k$ within segment

---

## Phase 3 — Data-driven changepoints (ruptures / PELT)

**Goal:** let the data decide where regimes change.

Use the PELT (Pruned Exact Linear Time) algorithm on the latent trajectory $Z \in \mathbb{R}^{n_t \times k}$:

- Cost model: `rbf` (kernel-based, robust to non-Gaussian noise)
- Model selection: sweep $K = 0 \ldots 6$ breakpoints, minimise BIC:

$$\text{BIC}(K) = \mathcal{C}(\hat{z}_{1:T}) + (K \cdot d_A + K) \log T$$

where $d_A = k^2$ is the number of parameters per $A_k$ matrix.

Compare data-driven breakpoints to the neural phase boundaries from Phase 2.

---

## Phase 4 — Eigenstructure across segments

For each fitted $A_k$, analyse eigenvalues $\lambda^{(k)}_j \in \mathbb{C}$:

| Property | Interpretation |
|----------|---------------|
| $|\lambda^{(k)}_j|$ | decay / growth rate per segment |
| $\arg(\lambda^{(k)}_j)$ | rotation frequency |
| $|\lambda| \approx 1$, $\arg \neq 0$ | sustained rotation (oscillation) |
| $|\lambda| < 1$, $\arg \approx 0$ | exponential decay |

Visualise eigenvalues in the complex plane per segment, overlaid on the unit circle.

---

## Phase 5 — Switching LDS (contingency)

If piecewise LDS open-loop R² is still <0.6 or the BIC-optimal $K > 4$, move to a proper
**Switching LDS (SLDS)**:

$$z_{t+1} = A_{s_t} z_t + \epsilon_t, \quad s_t \in \{1, \ldots, K\}$$

with a Markov switching prior on $s_t$, fitted via EM.  Implementation: `ssm` library
(Linderman et al.).

---

## Implementation

New module: `core/piecewise_lds.py`

| Function | Description |
|----------|-------------|
| `get_latent_trajectory(W, k)` | SVD → $Z$ (k, n_t), var explained, right singular vecs |
| `diagnose_lds_residuals(W, k)` | Fit global $A$, return per-step $\|z_{t+1}-Az_t\|^2$ |
| `fit_segment(Z, t_start, t_end)` | Fit $A_k$ on a slice, return $A$, one-step R² |
| `fit_piecewise_lds(W, breakpoints, k)` | All segments + open-loop rollout R² |
| `find_changepoints(W, k, n_bkps, model)` | PELT with fixed $K$ |
| `find_changepoints_bic(W, k, max_bkps)` | PELT + BIC sweep, returns best breakpoints |
| `compare_lds_vs_piecewise(W, breakpoints, k)` | Side-by-side metrics table |

---

## Expected Outcome

- Per-segment one-step R² should all be ≥ 0.99 (confirming locally linear).
- Open-loop R² should improve substantially over global LDS (0.24 → >0.6 target).
- PELT breakpoints should approximately align with the onset and plateau transitions
  (~50–80 ms and ~150–200 ms post-stimulus).
- Onset segment eigenvalues should show stronger rotation; sustained segment should show
  slower, more damped dynamics.
