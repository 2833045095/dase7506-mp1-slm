# DASE7506 — MP1 Small Language Model Challenge

Individual coursework submission. Train a language model from scratch on the
supplied WikiText-2 benchmark and achieve the lowest reproducible test bits per
byte (BPB) within the evaluation budget.

**Submitted result: 1.6603 test BPB** — the supplied baseline scores 2.1013
test BPB on its own 1,200-step recipe, and 1.8571 when trained to the same
budget as the submission.

## Repository layout

| Path | Contents |
|---|---|
| `guide/GUIDE.md` | Assignment guide. Provided, unmodified. |
| `code/README.md` | Technical instructions for the code package. Provided, unmodified. |
| `code/student.py` | **The submitted model** and its `build_model` factory. |
| `code/configs/student.json` | Configuration of the submitted model. |
| `code/configs/ablation-no-rope.json` | RoPE ablation (one key different). |
| `code/configs/ablation-no-dropout.json` | Dropout ablation (one key different). |
| `code/REPORT.md` | Report: method, comparisons, ablation, critical analysis. |
| `code/measure.py` | Evaluation-budget measurement helper (added by this submission). |
| `code/runs/` | Training outputs. Not tracked by git. |

The supplied files `common.py`, `evaluate.py`, `model.py`, `train.py`,
`data/`, `tests/` and `configs/baseline.json` are **unmodified**; their SHA-256
hashes still match `code/PACKAGE_MANIFEST.json`. `measure.py` and the two
student configs are the only additions to `code/`.

## Reproduction

Requires Python 3.12. All commands run from `code/`.

### 1. Install

```bash
cd code
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### 2. Evaluate the submitted checkpoint, without retraining

The inference assets are the checkpoint alone. Nothing else is required beyond
the code in this repository.

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --split test
```

The command prints the JSON result; `bpb` is the ranked score. It writes
`test_cpu_fp32.json` next to the checkpoint and a per-window loss file.

### 3. Retrain from scratch

```bash
python train.py --implementation student --config configs/student.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/final
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --split test
```

The checkpoint stores the implementation module name and the configuration, so
the evaluator rebuilds exactly the submitted predictor. Re-running the training
command re-seeds the initialisation and the data order, but GPU kernels differ
in their last bits, so retraining is expected to land within a small tolerance
of the reported score rather than bit-exactly. Scoring the frozen checkpoint in
FP32 on CPU is deterministic.

The controls reported in `REPORT.md` — the provided baseline, plus the RoPE and
dropout ablations, all at the same budget and seed:

```bash
python train.py --implementation model --config configs/baseline.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-baseline
python train.py --implementation student --config configs/ablation-no-rope.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-ablation
python train.py --implementation student --config configs/ablation-no-dropout.json \
    --device cuda --precision bf16 --seed 17 --steps 3000 --eval-every 500 --run-dir runs/s3-nodropout
```

The two ablation configs differ from `configs/student.json` in exactly one key
(`rope` and `dropout` respectively), so each command isolates one mechanism.

### 4. Measure the evaluation budget

Run once per checkpoint. Each invocation is a fresh process, so the reported
peak memory belongs to that checkpoint alone. Run the baseline and the
submitted checkpoint back to back on an idle machine; the time limit is a ratio
between the two.

```bash
python measure.py --checkpoint runs/s3-baseline/checkpoint.pt --split test
python measure.py --checkpoint runs/final/checkpoint.pt --split test
```

## Results

See [`code/REPORT.md`](code/REPORT.md) for the full study. Summary of the
controlled comparison, all runs at identical device, precision, seed and number
of processed training targets:

| Run | Test BPB |
|---|---|
| Provided baseline, supplied recipe (1,200 steps) | 2.1013 |
| Provided baseline, at the submission's budget (3,000 steps) | 1.8571 |
| Ablation: no RoPE | 1.7406 |
| Ablation: no dropout | 1.6755 |
| **Submitted model** | **1.6603** |

## AI assistance disclosure

In line with the assignment's AI-assistance policy:

* The **conceptual design, the implementation in `code/student.py`, the
  architectural choices, the experiment plan, the ablations and the analysis**
  were produced with substantive help from an AI coding assistant (Trae CN,
  GLM-5.3), working interactively with the author.
* The **provided baseline package was not modified**. Every file this
  submission depends on — `common.py`, `evaluate.py`, `model.py`, `train.py`,
  `tests/test_contract.py`, `configs/baseline.json`, `requirements.txt` and all
  of `data/` — still matches its SHA-256 in `code/PACKAGE_MANIFEST.json`, and
  `common.load_data()` still accepts the benchmark files against
  `code/data/manifest.json`. `student.py` differs by design: that is the
  submitted model.
* Two supplied **documents** do not match their recorded hashes in
  `code/PACKAGE_MANIFEST.json` as delivered: `guide/GUIDE.md` and
  `code/README.md`. No change was made to either of them here; they are not
  read by the evaluator, and no code path depends on them. The discrepancy is
  flagged rather than hidden so it can be checked against the released package.
* A `.gitattributes` file pins every path to byte-exact checkout (`* -text`).
  Without it, a Windows clone with `core.autocrlf` enabled would rewrite the
  line endings of `code/data/*` and the SHA-256 check inside
  `common.load_data()` would reject the benchmark as modified.
* **All reported numbers come from runs executed on the author's machine.** No
  number in `REPORT.md` or in this file is estimated, interpolated or copied
  from elsewhere. Every score can be reproduced with the commands above.
* The **test split was never used** for any design or selection decision.
  Architecture and step count were chosen on the validation split; the test
  split was evaluated only after the method was frozen.

## Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843);
the text is by Wikipedia contributors. The upstream dataset identifies
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html).
Retain these notices when redistributing the data. These notices do not assign
a new licence to the surrounding classroom code.