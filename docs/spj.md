# SPJ reflection, proposal, and judging

This repository includes the original SPJ reflection and proposal stages,
bounded candidate schema, native batch judge preparation and import, mirrored
comparisons, selection analysis, and lesson-register helpers. `spj_evolve.py`
runs the case and candidate sweeps through the portable agent API.

## Source and adaptation

The imported stages come from the original
[`TaiyuZhu/MND_agent` SPJ source](https://github.com/TaiyuZhu/MND_agent/tree/49caabc9f87744adca01ee02f1294fb824d44ffd)
on branch `spj`, commit `49caabc9f87744adca01ee02f1294fb824d44ffd`.
The retained stages include parent-specific transcript reflection, sequential
mutation proposals, authenticated imports, deterministic screening, blinded
mirrored judging, bootstrap selection analysis, and proposer-only development
lessons.

Provider settings are configurable. Candidate overlays run through the LangGraph
agent and web service. Supply a local case bank through `--case-dir`; case data,
experiment outputs, and logos are excluded from Git and source releases.

## Models and installation

```bash
python -m pip install "openai>=2.54,<3" "anthropic>=0.125,<1"
```

Export the model settings for your providers:

```bash
export MND_PROPOSAL_PROVIDER=anthropic
export MND_PROPOSAL_MODEL=your-proposal-model
export MND_PROPOSAL_API_BASE=https://api.anthropic.com
export MND_PROPOSAL_API_KEY=your-proposal-api-key
export MND_PROPOSAL_REASONING_EFFORT=high
export MND_PROPOSAL_MAX_TOKENS=8192

export MND_JUDGE_PROVIDER=openai
export MND_JUDGE_MODEL=your-judge-model
export MND_JUDGE_API_BASE=https://api.openai.com/v1
export MND_JUDGE_API_KEY=your-judge-api-key
export MND_JUDGE_REASONING_EFFORT=high
export MND_JUDGE_MAX_TOKENS=8192
```

The original proposer uses native Anthropic Messages. Choose a compatible model
and its supported reasoning effort; the model name is unrestricted. Native
judges support OpenAI Responses and Anthropic Messages, including their batch
APIs. To switch a judge to Anthropic, change its provider, endpoint, model, and
credentials together. A service that offers only `/chat/completions` cannot run
the native SPJ batch stages.

`MND_AGENT_*` independently configures the counselling model used during sweeps.
The native helpers also accept `MND_PROPOSAL_TIMEOUT` and `MND_JUDGE_TIMEOUT`.
Preparation flags such as `--model`, `--provider`, `--effort`, and
`--max-output-tokens` override defaults where shown by each command's `--help`.
Prepared manifests freeze model and endpoint metadata; keys remain in the
environment and are selected by the recorded `model_role`.

## 1. Run a baseline sweep

Provide a case bank with `--case-dir`. A fresh clone or source ZIP does not
include `research/spj_v1/cases/` or `vignettes/`. The case loader validates the
synthetic case format and its content hashes; it does not create missing data.
The default local path remains `research/spj_v1/cases/` for existing local runs.

Validate inputs first. This command calls no service and writes no output:

```bash
python spj_evolve.py --case-dir /path/to/local_cases \
  --cases all --output-dir .runtime/spj/baseline --plan-only
```

Stop any service already using port 8000, then start an offline evaluation
service in one terminal:

```bash
MND_ALLOW_PROMPT_CANDIDATES=1 MND_RECORD_EFFECTIVE_PROMPTS=1 python agent_api.py
```

In a second terminal using the same environment:

```bash
python spj_evolve.py --case-dir /path/to/local_cases \
  --cases all --output-dir .runtime/spj/baseline
```

The runner sends the fixed patient turns to the configured agent, records
transcripts and state, authenticates delivered candidate instructions, and
writes the original-compatible sweep layout. `--case-dir` selects another
valid synthetic case bank. Use a new output directory for each run.

`parent_inventory/summary.json` is an **unscored candidate inventory** used to
identify the baseline parent. It is not a judge score or selection result.
Changing `--cases` is useful for a smoke test; reflection preparation must then
use the matching `--expected-parent-case-count` instead of its default 24.

## 2. Prepare and import reflections

```bash
python propose_prompt.py prepare-reflections \
  --parent-analysis-dir .runtime/spj/baseline/parent_inventory \
  --parent-sweep-dir .runtime/spj/baseline \
  --parent-id baseline --generation 1 --candidate-prefix spj-g1 \
  --output-root .runtime/spj/proposals-g1
```

This writes `reflection-batch-prepared/` under the output root. It prepares
requests without submitting them. Use the batch steps below to count, submit,
and retrieve this batch, saving the result as
`.runtime/spj/proposals-g1/reflection-results.jsonl`. Then import it:

```bash
python propose_prompt.py import-reflections \
  --output-root .runtime/spj/proposals-g1 \
  --results .runtime/spj/proposals-g1/reflection-results.jsonl
```

Imports require the prepared request IDs and hashes to match. The
`prepare-corrective-reflections` stage and repeated `--corrective-results`
imports support failed or truncated reflections without discarding the original
attempt records.

## 3. Generate candidates sequentially

For an initial six-candidate generation from the baseline:

```bash
python propose_prompt.py prepare-proposal \
  --output-root .runtime/spj/proposals-g1 \
  --parent-sweep-dir .runtime/spj/baseline \
  --candidate-index 1 --offspring-per-parent 6
```

This prepares `proposal-batches/spj-g1-c01-prepared/`. Count, submit, and retrieve
that batch, then import its result:

```bash
python propose_prompt.py import-proposal \
  --output-root .runtime/spj/proposals-g1 \
  --parent-sweep-dir .runtime/spj/baseline \
  --candidate-id spj-g1-c01 \
  --results .runtime/spj/proposals-g1/spj-g1-c01-results.jsonl
```

Repeat preparation, execution, and import for indexes 2 through 6, using matching
`c02` through `c06` IDs and result paths. Candidate N sees the previously
imported candidates, so these proposal requests must remain sequential. One or
two parents are supported; the later-generation default is three offspring per
parent.

Each candidate carries its parent, generation, rationale, provenance, and hash.
Mutations target `system`, `background`, `education`, `questions`, or `wrap_up`.
The current runtime validates at most two directives of at most 500 characters
each; the imported cumulative candidate must also satisfy that bound.

After all six imports and their accounting records are complete:

```bash
python propose_prompt.py finalize \
  --output-root .runtime/spj/proposals-g1 --offspring-per-parent 6
```

The finalized archive records candidates as proposed and not yet evaluated.

## Batch counting, submission, and retrieval

The same native manager transports proposal batches and judge batches. For
each prepared batch, set `BATCH_DIR` to its directory. Supply current model
prices and their source in the named shell variables before counting:

```bash
python -m experiments.manage_frontier_pairwise_batch count \
  --batch-dir "$BATCH_DIR" \
  --standard-input-price-per-million "$INPUT_PRICE_PER_MILLION" \
  --standard-output-price-per-million "$OUTPUT_PRICE_PER_MILLION" \
  --batch-price-multiplier "$BATCH_PRICE_MULTIPLIER" \
  --pricing-source-url "$PRICING_SOURCE_URL" \
  --pricing-checked-on "$PRICING_CHECKED_ON"
```

Counting contacts the provider's token-count API without generating outputs.
Review `token_count_and_cost.json`. The original submission gate requires that
authenticated cost record and an explicit paid-submission argument:

```bash
python -m experiments.manage_frontier_pairwise_batch submit \
  --batch-dir "$BATCH_DIR" --confirm-paid-submission I_APPROVE_PAID_BATCH
```

Keep the returned batch ID and retrieve the completed result to the result
path used by the corresponding import command:

```bash
python -m experiments.manage_frontier_pairwise_batch retrieve \
  --batch-dir "$BATCH_DIR" --batch-id "$BATCH_ID" --output "$RESULTS_PATH"
```

The manager infers proposal credentials from proposal manifests and judge
credentials from judge manifests. `--role proposal` or `--role judge` provides
an explicit override. A retrieve command reports an unfinished batch's status;
retrieve again after completion. Existing receipts and cost records are
preserved rather than overwritten.

## 4. Sweep and judge the proposed generation

With the offline agent service still running:

```bash
python spj_evolve.py --case-dir /path/to/local_cases --cases all \
  --candidate-dir .runtime/spj/proposals-g1/candidates \
  --output-dir .runtime/spj/g1-sweep
python -m evaluation.build_spj_generation_pairs \
  --sweep-dir .runtime/spj/g1-sweep --output-dir .runtime/spj/g1-pairs
python -m experiments.manage_frontier_pairwise_batch prepare \
  --presentations .runtime/spj/g1-pairs/presentations.jsonl \
  --output-dir .runtime/spj/g1-judge-prepared --judge-id configured-judge
```

The baseline is included automatically. Pair construction randomizes system
identity and presents both conversation orders. Keep `allocation.key` separate
from judge inputs. Count, submit, and retrieve the judge batch using the steps
above, then import:

```bash
python -m experiments.manage_frontier_pairwise_batch import \
  --provider openai --batch-dir .runtime/spj/g1-judge-prepared \
  --results .runtime/spj/g1-judge-results.jsonl \
  --output-dir .runtime/spj/g1-judge
```

Use `--provider anthropic` for a batch prepared for Anthropic. A separate
`experiments.manage_frontier_pointwise_batch` manager provides absolute rubric
scoring for prepared pointwise `--items`, with matching preparation, counting,
submission, retrieval, corrective, and import stages.

## 5. Analyze and review selection

For a single native judge, matching the single-judge analysis setting:

```bash
python -m evaluation.analyze_spj_generation \
  --presentations .runtime/spj/g1-pairs/presentations.jsonl \
  --allocation-key .runtime/spj/g1-pairs/allocation.key \
  --judge-output-dir .runtime/spj/g1-judge \
  --sweep-dir .runtime/spj/g1-sweep --output-dir .runtime/spj/g1-analysis \
  --minimum-judges 1
```

The analysis defaults to requiring two completed judges unless explicitly
overridden. It checks completeness, averages mirrored orders, clusters by case,
and reports eligibility under its selection rules. Inspection of the underlying
conversations and critical-error nominations remains part of reviewing a
candidate. The analysis output is the scored parent summary for a later round;
it is distinct from the baseline's unscored inventory.

## Later rounds and accumulated lessons

The next round uses the scored `g1-analysis` summary and the corresponding
`g1-sweep` as its parent inputs. Choose one or two reviewed parent IDs from that
analysis and pass them with repeated `--parent-id` options when preparing the
next reflections:

```bash
python propose_prompt.py prepare-reflections \
  --parent-analysis-dir .runtime/spj/g1-analysis \
  --parent-sweep-dir .runtime/spj/g1-sweep --parent-id "$PARENT_ID" \
  --generation 2 --candidate-prefix spj-g2 \
  --output-root .runtime/spj/proposals-g2
```

After judging a round, freeze the accumulated development lessons. With the
directory names used in this guide, a first-round snapshot is:

```bash
python - <<'PY'
from pathlib import Path
from experiments.spj_lesson_register import freeze_register

freeze_register(
    Path('.runtime/spj/lessons-g1.json'),
    [(1, Path('.runtime/spj'))],
)
PY
```

After the next round's reflections have been executed and imported, include
that frozen snapshot in each proposal preparation:

```bash
python propose_prompt.py prepare-proposal \
  --output-root .runtime/spj/proposals-g2 \
  --parent-sweep-dir .runtime/spj/g1-sweep --candidate-index 1 \
  --lesson-register .runtime/spj/lessons-g1.json
```

Use the same snapshot for every candidate in that round. For later snapshots,
include all completed `(generation, lineage_directory)` pairs in
`freeze_register` and preserve earlier files. The register binds its content to
a hash; it records judge nominations as unverified repair signals and candidate
instructions as observed associations. Lessons provide proposer context and
are never injected into the clinical agent automatically.

## Use a reviewed candidate in the web app

Stop the offline service, then start the web app with the selected candidate:

```bash
MND_PROMPT_CANDIDATE=.runtime/spj/proposals-g1/candidates/spj-g1-c01.json bash start_app.sh
```

The backend and web interface use the selected candidate for their sessions.
Leave `MND_ALLOW_PROMPT_CANDIDATES` and `MND_RECORD_EFFECTIVE_PROMPTS` unset for
ordinary web use. These are separate offline sweep controls, and candidate
generation or finalization does not activate a candidate by itself.
