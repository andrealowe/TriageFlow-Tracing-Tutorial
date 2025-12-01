# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- Write concise, clear code
- Keep comments brief and relevant to the code itself
- Avoid verbose or explanatory comments

## Project Overview

TriageFlow is a multi-agent incident triage and response system demonstrating Domino's GenAI tracing capabilities. It showcases two key differentiators:
1. **Inline Evaluation** — Evaluators defined directly in the `@add_tracing` decorator
2. **Aggregated Metrics** — Declarative metric aggregation at run time using tuples

## Architecture

The system processes incidents through a sequential pipeline of four specialized agents:
1. **ClassifierAgent** — Categorizes incidents and assigns urgency (1-5)
2. **ImpactAssessmentAgent** — Evaluates blast radius, affected users/systems, financial exposure
3. **ResourceMatcherAgent** — Identifies appropriate responders and validates SLA requirements
4. **ResponseDrafterAgent** — Generates stakeholder communications and action plans

Each agent has dedicated tools (in `tools/`) and inline evaluators for quality metrics.

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main triage pipeline
python main.py

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_agents.py

# Run a single test
pytest tests/test_agents.py::test_classifier_agent
```

## Environment Variables

```
OPENAI_API_KEY=<your-api-key>
```

## Key Dependencies

- `domino-python-sdk` (from git: https://github.com/dominodatalab/python-domino.git@master)
- `mlflow==3.2.0`
- `openai>=1.0.0`
- `pydantic>=2.0.0`

## Domino SDK Tracing Patterns

### Inline Evaluation
```python
@add_tracing(
    name="agent_name",
    autolog_frameworks=["openai"],
    evaluator=evaluator_function  # Returns dict of metrics
)
async def method(self, ...):
    ...
```

### Aggregated Metrics
```python
with DominoRun(
    agent_config_path=CONFIG_PATH,
    aggregated_metrics=[
        ("metric_name", "mean"),
        ("metric_name", "stdev"),
    ]
) as run:
    ...
```

## Project Structure

```
├── tracing-tutorial.ipynb  # Main notebook
├── config.yaml             # Prompts, tool schemas, model configs
├── src/
│   ├── __init__.py
│   ├── models.py        # Pydantic models
│   ├── agents.py        # Agent functions with LLM tool calling
│   ├── tools.py         # Tool implementations
│   └── judges.py        # LLM judge evaluators
└── example-data/
    ├── financial_services.csv
    ├── healthcare.csv
    ├── energy.csv
    └── public_sector.csv
```

## Tracing Pattern (Single Line)

The key pattern from domino-genai-instrumentation-example:

```python
@add_tracing(name="triage_incident", autolog_frameworks=["openai"])
def triage_incident(incident: Incident):
    # All LLM calls inside are automatically captured with span types
    ...
```

- ONE `@add_tracing` decorator on the main function
- `autolog_frameworks` enables MLflow autolog for that provider
- Autolog captures all LLM calls as child spans with proper span types
- No manual spans needed

## Current Session State

- Provider selection: OpenAI or Anthropic via dropdown
- Experiment naming: `tracing-{username}`
- Run naming: `workspaceSession-{username}-{timestamp}`
- Evaluator: `pipeline_evaluator` extracts metrics from outputs
- Aggregated metrics: classification_confidence, impact_score, resource_match_score, completeness_score

## Dockerfile Installation

```dockerfile
FROM python:3.10
RUN pip install "domino[agents] @ git+https://github.com/dominodatalab/python-domino.git@release-2.0.0"
ENV MLFLOW_TRACKING_URI=http://localhost:5000
```