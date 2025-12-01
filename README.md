# TriageFlow

A multi-agent system that automatically triages incoming incidents—security alerts, service disruptions, compliance issues, or operational failures—by classifying them, assessing impact, assigning responders, and drafting stakeholder communications.

Demonstrates Domino's GenAI tracing and evaluation capabilities through a realistic, production-style pipeline applicable across financial services, public sector, healthcare, and critical infrastructure.

## Pipeline

Incidents flow through four specialized agents:

1. **ClassifierAgent** — Categorizes the incident and assigns urgency
2. **ImpactAssessmentAgent** — Evaluates blast radius, affected users, and financial exposure
3. **ResourceMatcherAgent** — Identifies available responders based on skills and SLA requirements
4. **ResponseDrafterAgent** — Generates communications tailored to each stakeholder audience

Each agent uses dedicated tools to query historical data, check resource availability, and apply organizational policies.

## Key Features

**Inline Evaluation** — Evaluators defined directly in the `@add_tracing` decorator enable real-time quality assessment without a separate evaluation pipeline.

**Aggregated Metrics** — `DominoRun` captures statistical summaries (mean, median, stdev) across all traces for monitoring classifier confidence, impact scoring consistency, and processing latency.

**Configuration-Driven Architecture** — System prompts, model settings, and agent parameters are centralized in `config.yaml`.

## Setup

1. Save your API key as a Domino user environment variable:
   - Go to **Account Settings** → **User Environment Variables**
   - Add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open `demo.ipynb` and run the cells sequentially. Select your provider (OpenAI or Anthropic) from the dropdown, then execute the triage pipeline.

View traces in the Domino Experiment Manager.

## Files

| File | Description |
|------|-------------|
| `demo.ipynb` | Interactive notebook walkthrough |
| `config.yaml` | Model and agent configuration |
| `agents.py` | Four triage agents |
| `models.py` | Pydantic data models |
| `tools.py` | Agent tool functions |
