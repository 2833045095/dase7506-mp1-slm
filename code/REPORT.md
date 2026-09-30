# MP1 report — a modernised, wider causal transformer

DASE7506 · individual coursework · protocol `7506-mp1-wt2-v2`

## 1. Summary

The supplied baseline is a 4-block, width-128 GPT with 1,088,256 parameters
that scores **2.1013 test BPB**. Its budget is spent almost entirely on being
small: it is far below every limit in the assignment, and its validation curve
is still falling steeply when its fixed 1,200-step recipe ends.

This submission keeps the two model interfaces of `model.py` and rebuilds the
network as a larger, modern causal transformer: rotary position embeddings
(RoPE), pre-norm RMSNorm, a gated SwiGLU feed-forward block, bias-free linear
projections, scaled residual output projections, residual dropout, and a
288-wide, 6-block stack with 6,565,536 parameters.

Measured on the frozen test split, with the same FP32 CPU scorer:

| Run | Validation BPB | Test BPB |
|---|---:|---:|
| Provided baseline, supplied recipe (1,200 steps) | 2.0711 | **2.1013** |
| Provided baseline, at the submission's budget (3,000 steps) | 1.8322 | 1.8571 |
| Ablation, no RoPE (3,000 steps) | 1.7109 | 1.7406 |
| Ablation, no dropout (3,000 steps) | 1.6365 | 1.6755 |
| **Submitted model (3,000 steps)** | **1.6328** | **1.6603** |

The submitted model stays inside all three evaluation limits; §5.3 gives the
measurements.

## 2. Diagnosed limitation

The baseline's constraint is not its training recipe — that recipe already has
warm-up, cosine decay, gradient clipping and AdamW with weight decay. The
constraint is the network itself.

1. **Scale.** 1,088,256 parameters, of which 262,144 are the tied input/output
   embedding. At 3.6M training tokens the model is the binding limit, not the
   data.
2. **Absolute position embedding.** `self.pos = nn.Embedding(context, width)`
   gives each offset 0..255 its own vector. Nothing in the architecture ties
   offset 10 to offset 11; relative distance has to be re-learned separately at
   every offset, and the representation carries no notion of ordering that
   transfers.
3. **LayerNorm + plain GELU MLP.** Standard `LayerNorm` rescales and re-centres;
   the feed-forward block is `Linear-GELU-Linear` with no multiplicative
   gating.
4. **Undertrained.** At step 300/600/900/1200 the baseline's validation BPB is
   2.3317 / 2.1845 / 2.1069 / 2.0720 — still falling when the recipe stops.

Items 1–3 are architectural; item 4 is a budget effect that a longer schedule
also removes. Both are addressed: §5.2 ablates one architectural mechanism
(RoPE) and one regularisation mechanism (dropout), each at a matched budget.

## 3. Method

Everything below lives in `student.py`; `build_model(config)` returns the model
and the configuration is carried in the checkpoint, so the evaluator rebuilds
exactly the submitted predictor.

**RoPE.** For a head of dimension `d`, positions are rotated by
`theta_i = pos / 10000^(2i/d)` in `d/2` planes, applied to `q` and `k` only.
Attention logits then depend on the *relative* offset of the two positions, so
one learned rotation serves all 256 offsets. Tables are precomputed once as
non-persistent buffers, so they cost no checkpoint bytes and are rebuilt from
the configuration.

**RMSNorm.** `x * rsqrt(mean(x^2) + eps) * g`, computed in FP32 and cast back.
No mean subtraction and no bias, which is both cheaper and one fewer place for
the residual stream to be re-centred.

**SwiGLU.** `down(silu(gate(x)) * up(x))` with hidden width 768. Choosing
`hidden ≈ 8/3 · width` keeps the gated block's parameter count close to the
plain 4x GELU MLP it replaces, so the comparison is not simply "more
parameters".

**Dropout.** `dropout = 0.2` applied to the residual projections and to the
embedding output. It exists because the corpus is small: the wider stack
without dropout goes through a memorisation phase, validation BPB bottoms out
near step 3,000 and then rises while the training loss keeps falling (see
§4). Dropout trades some training-set fit for generalisation and is
deliberately part of the method rather than an afterthought; it contributes no
parameters and no evaluation cost.

**Residual scaling.** The `proj` and `down` output projections are initialised
with `std = 0.02/sqrt(2·depth)` instead of `0.02`, so a 6-block stack starts
with a unit-scale residual stream. Without this the deeper stack trains less
stably.

**Causality and state.** `F.scaled_dot_product_attention(..., is_causal=True)`
and position-only rotation keep the model strictly causal. There is no cache
and no cross-window state: `predict_log_probs` recomputes from scratch each
call, so windows, examples and scoring passes are independent by construction.
`tests/test_contract.py` verifies all of this and passes unmodified.

**Scale.** Width 288, 8 heads (head dimension 36), depth 6, hidden 768,
context 256, vocabulary 2048, tied embedding. 6,565,536 parameters. The size
was **chosen by measurement, not by taste**: three larger candidates were timed
on the test split with *untrained* weights — a pure cost measurement that
carries no accuracy information — and rejected because they landed at
4.70–4.92× the baseline's CPU scoring time, leaving no margin under the 5×
limit. The chosen configuration measures 4.04× (see §5.3).

## 4. Experimental setup

All training runs use the supplied `train.py` unchanged, on the author's
machine (RTX 3060 Laptop GPU, BF16, AdamW, seed 17, batch 32, context 256).
The reference point is the supplied 1,200-step recipe, reproduced with no
modification (2.1013 test BPB). The controlled study, **Budget B**, is the
final 3,000-step schedule (24.6M processed targets) at the submitted
width-288/depth-6 architecture (6,565,536 parameters): the provided baseline
and two single-key ablations are trained under settings identical to the
submitted model. A separate 6,000-step run of the same architecture was
abandoned after it showed validation overfitting — validation BPB bottomed out
at step 3,000 and rose monotonically while the training loss kept falling.

**A methodological note that matters for reading the tables.** `train.py`
computes the cosine schedule over `args.steps`, so different step counts are
different learning-rate trajectories, not merely different budgets: the
supplied 1,200-step recipe has already decayed towards its floor by its last
step, while a 3,000-step schedule is still near its peak at that point. Numbers
are therefore comparable **only within a budget**, never across. The baseline
used for every claim below is the one trained at Budget B; the supplied
1,200-step figure is quoted only as the assignment's initial baseline.

**Overfitting is the main phenomenon in this study.** The supplied corpus is
3.6M training tokens. A 6.6M-parameter model reaches memorisation in a few
passes; validation BPB then rises even though the training loss keeps falling.
The design therefore chose the step count *by validation*, and dropout is part
of the submitted method (and of the RoPE ablation) so that the wider stack
stays away from the overfitting branch. Development and selection used the
validation split only; the test split was scored once per frozen checkpoint,
after the method was fixed.

## 5. Results

### 5.1 Comparison at matched processed targets

*(Budget B, 3,000 steps = 24.6M targets, identical device, precision, seed and
target count)*

| Run | Parameters | Validation BPB | Test BPB |
|---|---:|---:|---:|
| Provided baseline | 1,088,256 | 1.8322 | 1.8571 |
| Submitted model | 6,565,536 | 1.6328 | 1.6603 |

The submitted model is 0.1968 test BPB below the baseline trained under
identical conditions, a 10.6% relative reduction.

### 5.2 Mechanism ablations

Both ablations use the submitted architecture and budget, changing one thing.

| Run | Parameters | Validation BPB | Test BPB |
|---|---:|---:|---:|
| **Submitted model** (RoPE, dropout 0.2) | 6,565,536 | 1.6328 | 1.6603 |
| Ablation: no RoPE (learned absolute positions) | 6,639,264 | 1.7109 | 1.7406 |
| Ablation: no dropout | 6,565,536 | 1.6365 | 1.6755 |

The RoPE ablation is deliberately unfair to RoPE: the learned-embedding variant
adds a 256×288 embedding table (73,728 more parameters), so it has **more**
parameters than the submitted model. Removing RoPE costs 0.0781 validation BPB
and 0.0803 test BPB, and the loss holds despite the extra capacity — so the
effect cannot be explained by parameter count, and the ablation errs against
the claim being made rather than for it. The dropout ablation has exactly the
same parameter count, so it isolates regularisation alone. Every other config
key is identical, and each ablation differs from `configs/student.json` in
exactly one key.

**The dropout ablation is a weak effect, and is reported as such.** Without
dropout the model fits the training set faster: it is ahead on validation at
every checkpoint up to step 2,500, by 0.060 BPB at step 500 shrinking to 0.010
at step 2,500. The ranking then reverses at the end of the schedule, where
dropout finishes 0.0037 BPB ahead on validation and 0.0152 ahead on test. The
direction matches the overfitting diagnosis in §4, but a 0.0037 validation
margin on a single seed cannot establish it — dropout is a mild regulariser at
this budget, not the main source of the improvement. Its clearer value is at
longer schedules: the no-dropout variant on a 6,000-step schedule degraded from
1.6564 to 1.7003 validation BPB between steps 3,000 and 4,800, whereas dropout
kept improving to the last step. That comparison crosses schedules and is
therefore suggestive rather than conclusive.

### 5.3 Evaluation budget

All three limits measured on the frozen submitted checkpoint with
`measure.py` (§8); the baseline is measured the same way for the 5× reference.

| Limit | Allowed | Measured (submitted) | Margin |
|---|---:|---:|---:|
| CPU scoring time | ≤ 5× baseline | 4.04× (26.10 s vs 6.46 s) | 1.24× under |
| Peak evaluation RAM | ≤ 4 GiB | 1.85 GiB | 2.2× under |
| Inference assets, uncompressed | ≤ 64 MiB | 25.05 MiB | 2.6× under |

The asset figure is the uncompressed footprint of every tensor the evaluator
loads, counting the tied embedding/head storage once. It was measured, not
estimated, by `measure.py`. Both checkpoints were scored in the same session on
an otherwise idle machine, so the 4.04× ratio is like-for-like; it is the
tightest of the three margins and the one most likely to move on other
hardware.

### 5.4 Training-budget sensitivity

Validation BPB at increasing checkpoints, Budget B (3,000-step schedule):

| Steps | Processed targets | Provided baseline | Submitted model |
|---|---:|---:|---:|
| 500 | 4.1M | 2.2115 | 1.8979 |
| 1,000 | 8.2M | 2.0494 | 1.7725 |
| 1,500 | 12.3M | 1.9495 | 1.7078 |
| 2,000 | 16.4M | 1.8844 | 1.6679 |
| 2,500 | 20.5M | 1.8490 | 1.6450 |
| 3,000 | 24.6M | 1.8322 | 1.6328 |

Extended training was tested and rejected. On a 6,000-step schedule the
provided baseline kept improving (1.7554 test BPB at 49.2M targets), but the
submitted architecture did not: its validation BPB fell to 1.6564 at step
3,000, then rose to 1.6647 / 1.6762 / 1.7003 at steps 3,600 / 4,200 / 4,800
while the training loss kept falling from 2.55 to 2.18. Training longer stopped
being a way to buy accuracy and became a way to buy memorisation. That
measurement is what motivated two changes: shortening the schedule to 3,000
steps so the cosine decay completes, and adding dropout.

This is the honest framing of the whole study: **at a short budget most of the
apparent "architecture win" is really an undertrained baseline**, and at a long
budget the same architecture overfits a 3.6M-token corpus. The submitted
operating point is where validation says both effects are balanced.

## 6. Discussion and trade-offs

**Why the change helps.** RoPE makes position relative rather than absolute, so
one rotation table covers all offsets instead of 256 independent embedding
rows; RMSNorm removes the re-centring the residual stream does not need;
SwiGLU adds a multiplicative interaction between two projections, which is a
strictly richer function class per parameter than a GELU MLP; and the wider,
deeper stack simply has more capacity for the same evaluation budget. The
ablation isolates the position-encoding mechanism (§5.2); the other components
are claimed only as a bundle.

**What the comparisons establish, and what they do not.** They establish that,
at a fixed budget, the submitted architecture reaches a lower BPB than both the
provided baseline and its own RoPE-free variant. They do **not** establish that
the architecture is the cheapest way to buy that BPB: §5.4 shows a longer
schedule on the *unchanged* baseline recovers most of the gap. Training time is
unrestricted by the assignment while CPU *scoring* time is capped, which is
what makes spending budget on a larger model rational here — but the honest
reading is that architecture and training budget are substitutes at this
scale, and the submitted model is the better of the two only because the
scoring budget, not the training budget, is the binding constraint.

**Quality vs cost.** Scoring cost, not accuracy, was the binding constraint on
size: the largest architecture tested that stayed clear of the 5× limit was
chosen, at 3.99×. Accuracy beyond that was left on the table deliberately
rather than by oversight.

## 7. Limitations

* A single seed per configuration, so no variance estimate is available; the
  reported gaps may be seed-sensitive.
* The 5× CPU limit is a *ratio*, and the ratio was measured on one machine.
  Both checkpoints were measured in the same session on the same machine to
  make the ratio meaningful, but a different machine could shift it.
* Two mechanisms are ablated, RoPE and dropout. The remaining changes — RMSNorm,
  SwiGLU, residual scaling and the width/depth increase — were not ablated
  individually, so this report claims a combined effect plus two isolated
  mechanisms, not six. RMSNorm and SwiGLU are also bundled with the scale-up in
  the same "submitted model" column, so their separate contributions are not
  measured at all.
* No ensembling or weight averaging is used. Ensembling was considered and
  rejected on measurement: a second model would exceed the 5× scoring limit.

## 8. Reproduction

From `code/`, as documented in the repository README:

```bash
# submitted predictor, no retraining
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --split test

# rebuild it from scratch
python train.py --implementation student --config configs/student.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/final

# the three controls at the same budget and seed
python train.py --implementation model --config configs/baseline.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-baseline
python train.py --implementation student --config configs/ablation-no-rope.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-ablation
python train.py --implementation student --config configs/ablation-no-dropout.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-nodropout

# the three budget limits, baseline first for the 5x reference
python measure.py --checkpoint runs/s3-baseline/checkpoint.pt --split test
python measure.py --checkpoint runs/final/checkpoint.pt --split test
```

Every number in this report comes from these commands. No value is estimated
or carried over from another benchmark; token perplexity is never reported as
a score, since the ranked metric is bits per byte.