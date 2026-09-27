# Offline contracts tool

An **offline design tool**. It composes the repo's assume-guarantee contracts
soundly with [Pacti](https://github.com/FormalSystems/pacti), and reports where
the two contract libraries disagree. It does not run on the robot and changes no
flight code.

Keep Pacti off the robot, for two reasons:

- pacti 0.3.1 needs NumPy >= 2.2.6 and Python >= 3.10. ROS 2 Humble ships apt
  NumPy 1.21.5, so the two conflict (A-E1).
- Composing online inside the control loop costs about 120–160 ms per call, and
  the result is discarded (K10).

Consume the composed bounds from `out/contracts.json` instead.

The tool is **value-agnostic**. It shows both libraries side by side and does not
pick a winner. Choosing which contract values are authoritative is the user's
decision (decision DC).

## What it reads

Both libraries are imported read-only from `src/`. `src` is added to `sys.path`
inside the tool only, and bytecode writing is switched off, so nothing is written
under `src/`.

- The framework: `src/contracts/contract_framework.py`, i.e.
  `HierarchicalContractMonitor().define_contracts()`, with `SimpleContract` and
  `LinearConstraint`.
- The Pacti library: `src/planning/pacti_contracts.py`, i.e.
  `PactiContractLibrary().contracts`.
- The third copy of the wind limits: the `wind_limits` dicts in
  `src/planning/horizon_planner.py` and `src/planning/integrated_planner.py`.
  These are local variables inside methods, so they are read from the source
  with `ast`.

Values come from the imported objects. File:line locations come from parsing the
same files with `ast`.

## What it does

1. It converts every framework `SimpleContract` into a Pacti
   `PolyhedralIoContract`. Assumption boxes become two inequalities each. A
   guarantee `y <= sum(c*x) + k` becomes one half-space over all its inputs.
   Every conversion choice is listed in the report's section 3.
2. It composes each controller pipeline with Pacti, for each library. The
   pipeline is sensors -> estimator -> controller -> actuators, as the
   framework's `compose_pipeline` defines it.
   - Framework: `GPS_IMU_Sensors -> EKF_Estimator -> <controller> -> Actuators`.
   - Pacti library: `(gps || imu) -> ekf -> <controller> -> actuator`.
3. It records two reference compositions that the repo itself performs.
   Neither is a sound, complete pipeline.
   - `SimpleContract.compose`, which pre-flight uses today. It drops the
     G1 => A2 obligation (F-A1-01, K05).
   - `PactiContractLibrary.compose_pipeline`, which leaves IMU out (F-A1-25).
4. It writes one reconciliation row per (component, variable,
   bound-or-coefficient). A small renaming layer lines up the two libraries'
   variable names (`model.ALIASES`). Comparing H-inf `stabilization_time`
   (framework) with H-inf `settling_time` (Pacti library) is an assumption
   carried over from the A1 audit's F-A1-09 table (row 50). The code does not
   establish it, because no code in the repo reads either variable. It affects
   4 rows.

## Run it

Use Python >= 3.10. The pins are pacti 0.3.1's own minimum numpy, scipy and
matplotlib versions, which install on 3.10 through 3.13. `out/` is
byte-identical on 3.10, 3.11, 3.12 and 3.13, and CI runs 3.10 and 3.11. Run
from the repo root:

```bash
python -m pip install -r tools/contracts_offline/requirements.txt   # in a venv
python -B tools/contracts_offline/run.py          # or: python -B -m tools.contracts_offline
python -B tools/contracts_offline/run.py --check  # exit 1 if out/ is stale; writes nothing
python -B -m pytest -p no:cacheprovider tools/contracts_offline/tests
```

`-B` (or `PYTHONDONTWRITEBYTECODE=1`, as CI sets) keeps Python from leaving
`__pycache__` folders under `tools/`. They are gitignored, but without `-B` the
commands write more than `out/`.

A run takes about 7 s. Without pacti, the tool exits with code 2 and prints the
install command. The tests skip.

## Outputs (`out/`, committed)

- **`contracts.json`**: versioned by `schema_version`. It contains:
  - `provenance`: `inputs_sha256`, a content hash of the four input files with
    CRLF normalised to LF, plus a hash per file. No git commit ID is recorded,
    so cherry-picks, rebases and squashes do not change `out/`. The last commit
    touching the inputs is printed to stdout only.
  - `environment`: the pacti, numpy and scipy versions.
  - Every normalised bound (`bounds`), each with its file:line.
  - Per-contract input bounds (`contracts`).
  - The composed results per controller (`composed`). Each record carries:
    - a `source`: `framework` or `pacti_library`
    - a `method`: `pacti.compose`, or one of the two reference methods
    - the composed assumption terms (`assumptions`, and the multi-input ones
      again as `coupled_assumptions`), each meaning
      `sum(coefficients[v] * v) <= constant`
    - the envelope, and whether it equals the admissible set
      (`envelope_is_admissible_set`), with box corners that disprove it
      (`box_counterexamples`)
    - the composed output bounds
  - The third copy (`third_copy`), the reconciliation counts, and the
    conversion notes.
  - No field selects an authoritative value.
- **`reconciliation.md`**: the report.
  - The composed per-controller envelopes come first, side by side.
  - Then the counts, the conversion notes, how to read the rows, and the rows.
  - Paths are abbreviated: cf, pc, hp and ip.
- **`reconciliation.csv`**: the same rows, with full paths. It has these
  columns:
  - the framework value with its file:line
  - the named constant it comes from (`via`)
  - the Pacti-library value with its file:line
  - the third-copy value where one exists
  - match or mismatch
  - the row of A1's F-A1-09 table the item belongs to
  - the effect on the composed assumptions, for each library

**The envelope is a per-input projection, not the admissible set.** Each range
is that input's minimum and maximum with every other input free. When a record
has coupled assumptions, the box is larger than what the composition admits.
For example, framework MPC admits wind 9.609375 only at hdop 0.5, and the box
corner (hdop 4.666666667, wind 9.609375) violates
`0.0525*gps_hdop + 0.08*wind_speed <= 0.795`. Check a point against the full
`assumptions`, never against the box. `admissible.py` does this with the
standard library only:

```bash
python tools/contracts_offline/admissible.py framework mpc gps_hdop=4.6 wind_speed=9.6 --partial
# VIOLATED  0.0525*gps_hdop + 0.08*wind_speed <= 0.795   (left side 1.0095 > 0.795)
# ... NOT ADMISSIBLE (exit 1). Without --partial every input is required (exit 2 otherwise).
```

The effect column says, under that library's own values:

- For an assumption on a top-level input: whether it **binds** in the composed
  envelope or is **slack**.
- For an assumption on an internal variable: whether it is **discharged**
  (upstream already keeps the variable inside it) or makes Pacti **cut** the
  inputs.
- For a guarantee: which of those downstream obligations it feeds.

The effect of swapping a single value to the other library's is not computed.

## Determinism

Two runs produce byte-identical files:

- Keys and lists are sorted.
- Numbers are rounded to 10 significant digits.
- Line endings are LF.
- There are no timestamps and no git commit IDs; the inputs are identified by
  their content hash only.

CI regenerates `out/` and fails if it differs from the committed copy. The
output depends only on the input files and the tool, not on git history. An
input edit and the regenerated `out/` can therefore go in the same commit, and
cherry-picks, rebases and squash merges leave `out/` valid.

## Files

- `run.py` / `__main__.py`: the entry points.
- `cli.py`: builds and writes the outputs.
- `extract.py`: read-only import and `ast` locations.
- `compose.py`: the Pacti conversion, composition and effect analysis.
- `reconcile.py`: the rows and the F-A1-09 cross-check.
- `render.py`: the JSON, CSV and Markdown writers.
- `admissible.py`: exact point check against a composed record (standard
  library only; also a CLI).
- `provenance.py`: the commit, hashes and versions.
- `model.py`: constants, variable aliases and number formatting.
- `.gitignore`: re-includes `out/*.json` and `out/*.csv`, which the root
  `.gitignore` ignores.
