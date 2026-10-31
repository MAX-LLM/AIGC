# MND Agent

This repository accompanies *A large language model agent for genetic
counselling in motor neurone disease* by Taiyu Zhu and colleagues.

The code includes the counselling agent, a web interface for pre-test discussions,
and **Safety-constrained Patient–Judge Evolution (SPJ-Evolve)** for refining
counselling instructions through reflection, prompt proposals, and mirrored
judging. You can configure the agent, proposer, and judge models independently.

```text
Cases → Agent conversations → Reflections → Prompt proposals → Mirrored judging
```

## Repository layout

```text
agent_api.py / ui_gradio.py / start_app.sh   API and web app
mnd_langgraph_agent.py / agent_base.py      Counselling graph and prompts
model_config.py / api_backend.py            Model configuration and inference
propose_prompt.py / spj_evolve.py           SPJ entry points
prompt_candidate.py                        Validated prompt overlays
experiments/ / evaluation/                  Reflection, proposal, and judge stages
knowledge_base/ / rubrics/ / research/config/  Reference material and policies
tests/ / docs/                             Essential tests and guides
```

## Installation

Use Python 3.11 or later. Install the application dependencies:

```bash
python -m pip install requests fastapi uvicorn "pydantic>=2,<3" \
  "langgraph>=1.2,<2" "gradio>=6,<7"
```

To run the SPJ workflow, also install the provider libraries:

```bash
python -m pip install "openai>=2.54,<3" "anthropic>=0.125,<1"
```

## Run the application

Set your provider's model ID, API endpoint, and API key, then start the application:

```bash
export MND_BACKEND=api MND_RETRIEVAL_MODE=lexical
export MND_AGENT_MODEL=your-agent-model
export MND_AGENT_API_BASE=https://api.openai.com/v1
export MND_AGENT_API_KEY=your-api-key
bash start_app.sh
```

Open the [web app](http://127.0.0.1:8080) or the
[API documentation](http://127.0.0.1:8000/docs). Press Ctrl+C to stop both services.
The settings above use lexical retrieval, which requires no embedding download.
Exported environment variables override settings in an optional local `.env` file.

| Role | Model | Endpoint | API key |
| --- | --- | --- | --- |
| Agent | `MND_AGENT_MODEL` | `MND_AGENT_API_BASE` | `MND_AGENT_API_KEY` |
| Proposer | `MND_PROPOSAL_MODEL` | `MND_PROPOSAL_API_BASE` | `MND_PROPOSAL_API_KEY` |
| Judge | `MND_JUDGE_MODEL` | `MND_JUDGE_API_BASE` | `MND_JUDGE_API_KEY` |

The agent connects to an OpenAI-compatible chat completions endpoint. The SPJ
proposer uses the Anthropic Messages API: set `MND_PROPOSAL_PROVIDER=anthropic`
and `MND_PROPOSAL_API_BASE=https://api.anthropic.com`. The judge supports OpenAI
Responses or Anthropic Messages; set `MND_JUDGE_PROVIDER` to `openai` or `anthropic`.
Restart the application after changing model settings. Local inference with
Transformers or vLLM and [hybrid retrieval](knowledge_base/README.md) are also supported.

## SPJ workflow

Check a local case bank and preview the run without making API calls:

```bash
python spj_evolve.py --case-dir /path/to/cases \
  --output-dir .runtime/spj/baseline --plan-only
```

Stop the application, then start the backend with SPJ evaluation enabled. In a
second terminal, export the same model settings and run the case bank:

```bash
MND_ALLOW_PROMPT_CANDIDATES=1 MND_RECORD_EFFECTIVE_PROMPTS=1 python agent_api.py
# Second terminal:
python spj_evolve.py --case-dir /path/to/cases --output-dir .runtime/spj/baseline
```

See the [SPJ guide](docs/spj.md) for reflection, sequential proposal generation,
batch judging, and candidate selection. To use a reviewed candidate, set
`MND_PROMPT_CANDIDATE` to its JSON file and restart the application.

Case data, logos, logs, and experiment results remain local and are excluded
from Git and release ZIPs.
