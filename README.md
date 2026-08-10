# Learning-Rule Transplants

**What happens to a published model when you replace its learning rule with a biologically plausible one?**

Five self-contained notebooks that take the local plasticity rules of the EIANN framework — Galloni, Peddada, Chennawar & Milstein (2026), *Cellular and subcellular specialization enables biology-constrained deep learning*, [Cell Reports 45:117159](https://doi.org/10.1016/j.celrep.2026.117159) · [Milstein-Lab/EIANN](https://github.com/Milstein-Lab/EIANN) — and drop them into four unrelated published models: an active-inference Pong agent, a 3→3 auto-associator, an information-theoretic plasticity estimator, a β-VAE rate–distortion study, and a generative cognitive map for planning.

The question in each case is the same one EIANN's Figure 5 asks, moved out of image classification: **once a local error signal exists at a synapse, which plasticity rules still descend the gradient — and does that even predict downstream task performance?**

---

## Design constraints shared by all five notebooks

Every notebook here follows the same rules, which is what makes them comparable:

- **Nothing is installed and no repository is cloned.** The parts of EIANN each experiment needs — populations, projections, Dale's law, relaxation dynamics, the dendritic compartment, the learning rules — are reimplemented from the published v1.0.0 source inside the notebook, typically in 250–600 lines of PyTorch. Names are kept identical to the originals so the correspondence is checkable line by line.
- **Runs top-to-bottom in a fresh Google Colab CPU runtime.** Dependencies are limited to what Colab preinstalls. The tensors are far too small for a GPU to help; leave the runtime on CPU.
- **A `QUICK` / small-config switch** at the top of each notebook trades fidelity for a few minutes of runtime. Full configurations are documented alongside.
- **Self-tests, not trust.** Where a reimplementation could plausibly diverge from the source, there is an assertion that checks it.
- **Negative results are reported as results.** Several of these transplants do not work, and the notebooks say so and instrument *why* rather than quietly reporting a low number. See [Caveats](#caveats-please-read-before-quoting-any-number).

---

## The notebooks

| Notebook | Host model | The question | Runtime (quick config) |
|---|---|---|---|
| [Active Inference & Intentional Behaviour](#1-active-inference--intentional-behaviour--pong) | Friston et al. 2023, Pong | Can local rules learn closed-loop control on three architectures (`ann`, Dale's-law `ei`, dendritic `ei_dend`)? | ~5–10 min CPU (~30–60 min full) |
| [Items or Relations?](#2-items-or-relations--auto-associators) | Krause & Reimann 2024, 3→3 auto-associator | Does a network store the training *items* or the *symmetry group* relating them — and does the learning rule decide? | Minutes (tiny networks) |
| [Local Redundancy](#3-local-redundancy--an-information-theoretic-plasticity-measure) | arXiv:2607.13432 | Does the local-redundancy plasticity estimator track trainability across rules, where cheap baselines don't? | A few min CPU; MNIST section much slower |
| [The Geometry of Efficient Codes](#4-the-geometry-of-efficient-codes--β-vae-rate-distortion) | `damat-le/geom_eff_codes`, β-VAE on *Corridors* | Do rate–distortion distortion signatures (specialization, orthogonalization) survive when the μ-head is trained by a local rule? | Several min for all 28 runs |
| [GCML × EIANN](#5-gcml--eiann--plasticity-rules-inside-a-generative-cognitive-map) | Lin, Yang, Zhao, Pezzulo & Maass 2026, NMI | Transplant rules into a planner: goal-directed graph sampling, grid-cell imagination, compositional tiling | ~5 min CPU (~30–60 min full) |

---

### 1 · Active Inference & Intentional Behaviour — Pong

`EIANN_Active_Inference_and_Intentional_Behaviour_...ipynb`

Reproduces the Pong experiment of Friston, Salvatori, Isomura, Tschantz, Kiefer, Verbelen et al. (2023), [arXiv:2312.07547](https://arxiv.org/abs/2312.07547) — a ball bouncing at 45° in a small box, a paddle with three control states — and trains the agent with nine EIANN rules across three architectures: unconstrained `ann`, Dale's-law `ei` with lateral inhibition and 12 relaxation steps, and `ei_dend` adding a dendrite-targeting interneuron population and top-down projections onto the apical compartment.

Two metrics, answering different questions: **action accuracy** against the oracle on held-out states (compared against the *majority-action* baseline, not 1/3), and **hit rate / rally length** in closed loop, which is what the paper actually reports.

**One deliberate departure from the paper.** The paper's observation is a single 30-pixel frame, disambiguated by its latent transition model over 40 states. Feed-forward networks are memoryless, so a single frame makes the target a non-deterministic function of the input and the task unlearnable in principle. `obs_mode='pixels'` therefore stacks the current *and previous* frame — the minimal change that makes the task well-posed. `obs_mode='compact'` offers a 5-number alternative.

**What it found.** Backprop reaches ≈0.87 action accuracy and plays perfectly on all three architectures; the Dale-constrained network matches the unconstrained one, so biological constraints cost little at this scale. **Every local rule sits at or below the majority-action baseline.** The diagnostics locate the failure precisely: the output readout is fine (≈0.58 local vs ≈0.62 Adam with hidden layers frozen), the dendrite is healthy after calibration (signed error, ~6× larger), but `gradient_alignment` — the cosine between the locally computed dendritic error and true backprop's — sits at ≈+0.05 to +0.13, essentially zero.

Part 8 argues why, specifically: under Dale's law the forward weights are sign-constrained and cannot rotate into agreement with the fixed top-down pathway, while the effective top-down matrix is mixed-sign by construction, so achievable alignment is structurally bounded. Learning the top-down weights (`Top_Down_Hebbian_Temporal_Contrast`, on by default) moves alignment from ≈0.03 to ≈0.13 — real, but not enough inside this training budget.

---

### 2 · Items or Relations? — auto-associators

`EIANN_Items_or_Relations_...ipynb`

Reproduces Krause & Reimann (2024), *Items or Relations — what do Artificial Neural Networks learn?* ([arXiv:2404.12401](https://arxiv.org/abs/2404.12401)), then extends it along a learning-rule axis. A 3→3 network y = φ(Wx) is trained to reproduce a set of binary items; afterwards, has it stored the **items** or the **relations** between them (their symmetry group Σ = {e, (32)})?

Six parts: the symmetry group, orbits and Σ-compatible weight structure derived **symbolically with SymPy** rather than asserted; reference SGD/Adam solutions reproducing the paper's published matrices to better than 10⁻³; a ~250-line `minieiann`; 8 rules × 3 activations × 2 training sets; then generalization tables, eigen-spectra, plane-attractor diagnostics and flow fields matching the paper's Fig. 1.

**What it found.** The rules split cleanly, and the split is not about biological plausibility — it is about **whether the update has a zero-error fixed point**. Error-corrective rules (`Backprop`, `Top_Layer_BP_like_2E`, `BTSP_19`) reach zero cost and land inside the analytical family with the eigenvalue-1 pair and a contracting third: a plane attractor, hence generalization to unseen items of span(X). Correlational rules (Hebbian, Oja, BCM and their nudged variants) do not — though the boundary isn't perfectly sharp: `Supervised_Hebb_WeightNorm_4` on X′ at `scale=1.0` does build a plane attractor.

Two measurements that contradicted the obvious guesses, and are shown rather than assumed: only *unsupervised* Hebbian genuinely diverges (to 10¹⁵² by 12k steps) — both BCM variants are self-stabilizing, which is what the sliding threshold is for; and weight-norm scale 1.0 beats the "natural" 1.5 everywhere. There's also a nice aside on BTSP's eligibility trace binding each item to the *previous* sample, which is harmless on one training set and creates forbidden off-diagonal weights on the other.

The synthesis: representing *relations* rather than *items* requires both an activation with a linear regime where the data live **and** a rule with a fixed point at zero error. Neither alone suffices.

---

### 3 · Local Redundancy — an information-theoretic plasticity measure

`EIANN_Local_Redundancy_...ipynb`

Measures the local-redundancy plasticity estimator of *Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization* (arXiv:2607.13432) across EIANN rules, on two architectures matched for parameter count, Dale's-law constraints, weight normalisation and seed — so differences are attributable to the rule.

This notebook is largely a **correctness audit**. Six substantive issues in the original draft are documented in a table at the top, each fixed and each checked by a numbered self-test. The most consequential: the estimator was differentiating w.r.t. `requires_grad` parameters, and since EIANN sets `weight.requires_grad = False` on every projection except under `Backprop.__init__`, every non-backprop rule's estimate was computed **over the biases only** — making every cross-rule comparison apples-to-oranges. Also: a per-sample expectation of squared norms was being computed as the square of a batch-averaged gradient; only the last `backward_steps` of the relaxation were tracked; and the dendritic network had no inhibitory cancellation, so `dendritic_state ≥ 0` always and the "error" was just top-down drive.

Part 10 is a short guide to reading the results and the traps — absolute values aren't comparable across σ, ε or `n_probe`; `R**` scales with parameter count; Hebbian and Oja are unsupervised so low accuracy is expected and isn't evidence of lost plasticity; run more than one seed.

**Open caveat, stated in the notebook:** arXiv:2607.13432 could not be retrieved during writing, so the estimator is implemented from the formulas stated in the original notebook plus standard Fisher-information theory. Check §3.6 of the paper for the exact probe construction and σ/ε conventions before quoting absolute numbers.

---

### 4 · The Geometry of Efficient Codes — β-VAE rate–distortion

`EIANN_The_Geometry_of_Efficient_Codes_...ipynb`

Trains β-VAEs on the *Corridors* dataset under capacity constraints and asks whether the paper's distortion signatures — specialization by frequency class, orthogonalization by task class — survive when the encoder's `hidden → μ` layer is trained by a local rule instead of Adam. Two experiments: capacity × data distribution (28 models), and capacity × task (reconstruction-only vs β-VAE + linear classifier hybrid).

This is v3 of the notebook, and the version history is instructive. v2 was checked line-by-line against both source repos and fixed the *Corridors* generator (peak-normalized Gaussian bump, correct polarity), the β-VAE loss (MSE not BCE, the real Burgess capacity-annealing schedule with its `/4` damping, the separate fixed `kld_weight`, the classifier curriculum that ramps in only after capacity finishes annealing), and made plastic updates genuinely sample-by-sample with persistent per-sample buffers. v3 then handles a crash: **BCM diverged to NaN specifically for the high-capacity classifier model** — a known failure mode, since its postsynaptic activity is unclamped in the real `BCM_4` and `theta` only tracks activity via a slow EMA, so the classifier's extra gradient at high capacity tipped it into runaway feedback. The fixes (weight clamp, grad clipping, early-stop-with-warning, NaN-safe probe embedding) are labelled as a **disclosed pragmatic substitute** for the repo's real per-projection `weight_bounds` mechanism, not a source-verified fix.

Section 9 keeps a standing list of remaining simplifications: dense MLP encoder rather than the real conv stack; single-pass `delta = -∂loss/∂μ` standing in for EIANN's two-phase recurrent dendritic signal (justified as the correct leading-order reduction for `Hebbian_Temporal_Contrast` per Eq. 27, and structurally analogous for `BP_like_2E`/`BTSP_19`); and only the designated plastic layer trained sample-by-sample.

---

### 5 · GCML × EIANN — plasticity rules inside a generative cognitive map

`GCML_EIANN_Learning_Rules.ipynb`

The most direct cross-pollination. The host is Lin, Yang, Zhao, Pezzulo & Maass (2026), *Neural sampling from cognitive maps enables goal-directed imagination and planning*, [Nature Machine Intelligence 8:1045–1065](https://doi.org/10.1038/s42256-026-01254-4) ([LH-cbicr/GCML](https://github.com/LH-cbicr/GCML)). GCML learns four linear projections **Q, V, W, G** by local online plasticity and never backpropagates — and every GCML prediction error has exactly the shape of an EIANN dendritic state. So substituting EIANN's rules for GCML's delta rules is precisely paper 2's experiment, transplanted into planning.

Three tasks: goal-directed sampling on a 32-node abstract graph (with a full per-rule logarithmic learning-rate search, always reporting each rule's *best* setting); imagined 2-D trajectories from a 1000-cell grid-cell map, including the never-visited lower-left quadrant that tests whether a rule preserves the linearity **W** needs to extrapolate; and NP-hard compositional silhouette decomposition, trained on 5 blocks and tested on 8.

**What it found** — four things, in rough order of interest:

1. **Paper 2's central split transfers to planning.** Error-referring rules track the baseline; unsupervised Hebbian and Oja fail everywhere with alignment angles *greater* than 90°; BCM lands near 90°, orthogonal to the gradient, exactly as reported. More usefully, **the alignment angle measured on a few training batches predicts downstream planning performance** — so paper 2's cheap diagnostic can screen plasticity rules for planning models without running the planner.
2. **BTSP transfers, but only if you import its biological preconditions too.** It needs the signed-synapse (Dale's law) constraint — implemented by splitting each presynaptic population into rectified positive and negative banks — and it needs divisive normalization of the presynaptic population, because GCML's embedding magnitudes are arbitrary and *shrink ~10× during training*, which lets BTSP's depression term swamp potentiation. The general point: **plasticity rules with hard-coded thresholds are not scale-free**, and porting them into a model with drifting representations requires an explicit normalization step. Biology has one; a naive port doesn't.
3. **Error-correcting rules are not always right.** GCML's own inverse model on the tiling task is *associative*, storing W[a] ← o_{t+1} − o_t in one shot. With 664 actions and a 100-dimensional state the map is massively rank-deficient; a delta rule converges to the least-squares solution, optimal in-distribution — but the planner applies **W** far outside it. Least-squares buys accuracy in-distribution at the cost of extrapolation. Where paper 2 evaluates on i.i.d. classification and "closer to the gradient" is essentially always better, GCML's premise is out-of-distribution generalization, and **the ranking can invert.**
4. **The full port is much harder than swapping one matrix.** When **Q** itself is built by a non-gradient rule, the inverse model is fitted to a moving, badly-conditioned target and errors compound.

An appendix sketches the deeper experiment: since GCML's projections are single and linear, what's transplanted here are paper 2's *synaptic* rules, not dendritic target propagation proper. Making the inverse model two-layer with dendrite-targeting interneurons would test the actual multi-layer credit-assignment claim.

---

## The rule library

Not every rule appears in every notebook, but the naming follows EIANN's source (`EIANN/rules/{hebbian,backprop_like,btsp}.py`) throughout.

| Class / key | Consults the target? | Needs dendrites? | Notes |
|---|---|---|---|
| `Backprop` | ✓ | – | Baseline; autograd through the unrolled relaxation. Not biologically plausible — the ML upper bound. |
| `Hebb_WeightNorm`, `_4` | ✗ | – | ΔW = η·post·preᵀ. **Unbounded on its own** — pair with `weight_constraint='normalize'`. |
| `Supervised_Hebb_WeightNorm_4` | ✓ | ✓ | Hebbian on post-nudge rates. This is GCML's own Eq. 14. |
| `Ojas_rule` (+ nudged variant) | ✗ / ✓ | – | ΔW = η(yx − y²W); self-normalising PCA. |
| `BCM_4`, `Supervised_BCM_4` | (indirect) | – | Sliding threshold θ → ⟨y²⟩/k. Self-stabilizing; refers to activity only. |
| `Top_Layer_BP_like` / `_2E` | ✓ | – | Delta rule at the output projection. |
| `BP_like_1E` | ✓ | ✓ | Dendritic target prop; error = *change* in dendritic state. |
| `BP_like_2E` / `BP_like_2L` (`LDS`) | ✓ | ✓ | Error read straight off the dendrite — requires Dend-I cancellation to be meaningful. |
| `Hebbian_Temporal_Contrast` (`TCH`) | ✓ | ✓ | Contrastive Hebbian (Xie & Seung 2003); contrast between sensory and supervised phases. |
| `BTSP_19` | ✓ | ✓ | Saturating, weight-dependent one-shot plasticity with second-long eligibility traces. |
| `Dendritic_Loss` | – | ✓ | Trains Dend-I → E(dend) to cancel top-down excitation. |
| `Top_Down_Hebbian_Temporal_Contrast` | – | ✓ | Trains top-down weights toward W_forwardᵀ. |

**The distinction that is easy to get wrong:** EIANN's supervised local rules receive no backpropagated error. The backward phase *nudges the output population toward the target* — `plateau = clamp(t − y, −1, 1)`, `y_nudged = φ(s + plateau)` — and the update is then a purely local product of pre- and post-synaptic quantities evaluated *after* the nudge. This is what makes `Supervised_Hebb_WeightNorm_4` supervised while `Hebb_WeightNorm` is not. Applying an unsupervised rule to the output projection leaves **no path at all** from the label to the weights; several of these notebooks exist partly because an earlier draft did exactly that.

---

## Running them

**Colab (recommended).** Open any notebook and run all cells. Replace `<user>/<repo>` below with your own once the repo is pushed:

```
https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/<notebook>.ipynb
```

Tested on Python 3.10–3.11. CPU only — the tensors are small enough that a GPU doesn't help, and the EI architectures train one sample at a time with multi-step settling, so the bottleneck is Python, not FLOPs.

The uploaded filenames are descriptive but long enough to be awkward in URLs and `git` output; short numbered names with the full title in the notebook's first cell (where it already is) reads better on GitHub. Keep executed outputs committed so the notebooks render with figures on GitHub without anyone running them.

---

## Caveats — please read before quoting any number

These notebooks are **instrumented starting points, not reproductions of the original papers' claims.**

- **Hyperparameter budget.** Galloni et al. tuned their models with [Nested](https://github.com/neurosutras/nested), a multi-objective optimiser, on a cluster; their published configs are literally named `..._complete_optimized`. The presets here come from hand sweeps over a handful of values, or (in the GCML notebook) a three-point logarithmic search per rule. Rules with more internal parameters — BTSP, BCM — are therefore under-tuned relative to LDS, and their numbers should be read as lower bounds. Widening the grids is the single highest-value change anyone could make.
- **A negative result here is not a negative result about the rule.** Where a rule fails, the notebooks try to say *where* credit assignment breaks — gradient-alignment cosines, dendritic-balance checks, frozen-hidden-layer controls — rather than reporting a low number and stopping. Read the diagnostic before drawing the conclusion.
- **Single-seed differences are not reliable.** Vary the network and data seeds together and plot the spread.
- **Quick configs are noisier than full ones.** The `QUICK` / small-`total_steps` paths exist so the notebooks finish in minutes; absolute values shift when you scale up.
- **Each notebook lists its own known simplifications** in a final section. Those lists are meant to be maintained, not decoration.

---

## Citing

If you use any of this, cite the underlying papers — the donor and whichever host applies.

```bibtex
@article{galloni2026eiann,
  title   = {Cellular and subcellular specialization enables biology-constrained deep learning},
  author  = {Galloni, Alessandro R. and Peddada, Aaron and Chennawar, Yash and Milstein, Aaron D.},
  journal = {Cell Reports},
  volume  = {45},
  pages   = {117159},
  year    = {2026},
  doi     = {10.1016/j.celrep.2026.117159}
}

@article{lin2026gcml,
  title   = {Neural sampling from cognitive maps enables goal-directed imagination and planning},
  author  = {Lin and Yang and Zhao and Pezzulo, Giovanni and Maass, Wolfgang},
  journal = {Nature Machine Intelligence},
  volume  = {8},
  pages   = {1045--1065},
  year    = {2026},
  doi     = {10.1038/s42256-026-01254-4}
}

@article{friston2023active,
  title   = {Active Inference and Intentional Behaviour},
  author  = {Friston, Karl and Salvatori, Tommaso and Isomura, Takuya and Tschantz, Alexander and Kiefer, Alex and Verbelen, Tim and others},
  journal = {arXiv preprint arXiv:2312.07547},
  year    = {2023}
}

@article{krause2024items,
  title   = {Items or Relations --- what do Artificial Neural Networks learn?},
  author  = {Krause, Renate and Reimann, Susanne},
  journal = {arXiv preprint arXiv:2404.12401},
  year    = {2024}
}

@article{localredundancy,
  title   = {Local Redundancy: An Information-Theoretic Measure of Plasticity from Synthetic Memorization},
  journal = {arXiv preprint arXiv:2607.13432}
}
```

The rate–distortion notebook builds on the `damat-le/geom_eff_codes` codebase; cite its associated paper if you use that notebook.

EIANN v1.0.0 is archived at [Zenodo 10.5281/zenodo.18602556](https://doi.org/10.5281/zenodo.18602556), with docs at <https://milstein-lab.github.io/EIANN>. The reimplementations here were transcribed from the published source; if you have the repository to hand, diffing it against the reimplementation cells is worth doing, since the docs build reflects `main` and a tagged release may differ.

---

## License

*Choose one and add a `LICENSE` file — MIT is the usual default for notebooks like these.* Note that the reimplementation cells transcribe function bodies from EIANN's published source, so check EIANN's own license terms before choosing something more permissive than theirs.
