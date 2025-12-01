import json
from models import (
    Incident, Classification, ImpactAssessment, ResourceAssignment,
    ResponsePlan, Resource, Communication, IncidentCategory
)
from tools import (
    lookup_category_definitions, lookup_historical_incidents, calculate_impact_score,
    check_resource_availability, get_sla_requirements, get_communication_templates,
    get_stakeholder_list
)

CLASSIFIER_PROMPT = """You are an incident classification specialist. Analyze the incident and return JSON with:
- category: one of [security, operational, performance, data_integrity, compliance, infrastructure, user_access]
- subcategory: specific subcategory
- urgency: 1-5 (5=critical)
- confidence: 0.0-1.0
- reasoning: brief explanation
- affected_domain: department/area affected

Urgency guidelines:
- 5: Active breach, complete outage, data loss in progress
- 4: Significant degradation, potential security incident
- 3: Moderate impact, workarounds available
- 2: Minor impact, single user/system
- 1: Informational, minor requests

Category definitions: {categories}

Incident: {incident}

Return only valid JSON."""

IMPACT_PROMPT = """You are an impact assessment specialist. Given the incident and classification, return JSON with:
- impact_score: 0.0-10.0
- affected_users_estimate: integer
- affected_systems: list of system names
- financial_exposure: estimated $ impact or null
- similar_incidents: list from historical data
- blast_radius: one of [contained, department, organization, external]
- reasoning: brief explanation

Historical incidents: {historical}
Classification: {classification}
Incident: {incident}

Return only valid JSON."""

RESOURCE_PROMPT = """You are a resource allocation specialist. Based on classification and impact, return JSON with:
- primary_responder: {{resource_id, name, role, skills, availability, match_score}}
- backup_responders: list of resources
- sla_target_hours: float
- sla_met: boolean
- escalation_path: list of roles/names
- reasoning: brief explanation

Available resources: {resources}
SLA requirements: {sla}
Classification: {classification}
Impact: {impact}

Return only valid JSON."""

RESPONSE_PROMPT = """You are a communications specialist. Create response plan as JSON with:
- communications: list of {{audience, subject, body, urgency_indicator}}
- action_items: list of strings
- estimated_resolution_time: string
- escalation_required: boolean
- completeness_score: 0.0-1.0

Stakeholders: {stakeholders}
Templates: {templates}
Incident: {incident}
Classification: {classification}
Impact: {impact}
Resources: {resources}

Return only valid JSON."""


def call_llm(client, provider: str, model: str, prompt: str, temperature: float = 0.3, max_tokens: int = 1000) -> str:
    """Unified LLM call for both providers."""
    if provider == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    else:  # anthropic
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(lines[1:-1])
    return json.loads(response)


def classify_incident(client, provider: str, model: str, incident: Incident, config: dict) -> Classification:
    """Classify the incident type and urgency."""
    categories = lookup_category_definitions()
    prompt = CLASSIFIER_PROMPT.format(
        categories=json.dumps(categories, indent=2),
        incident=incident.model_dump_json()
    )

    response = call_llm(client, provider, model, prompt,
                        config["agents"]["classifier"]["temperature"],
                        config["agents"]["classifier"]["max_tokens"])
    data = parse_json_response(response)
    return Classification(**data)


def assess_impact(client, provider: str, model: str, incident: Incident,
                  classification: Classification, config: dict) -> ImpactAssessment:
    """Assess the impact scope and severity."""
    historical = lookup_historical_incidents(classification.category.value, classification.subcategory)
    prompt = IMPACT_PROMPT.format(
        historical=json.dumps(historical, indent=2),
        classification=classification.model_dump_json(),
        incident=incident.model_dump_json()
    )

    response = call_llm(client, provider, model, prompt,
                        config["agents"]["impact_assessor"]["temperature"],
                        config["agents"]["impact_assessor"]["max_tokens"])
    data = parse_json_response(response)
    return ImpactAssessment(**data)


def match_resources(client, provider: str, model: str, classification: Classification,
                    impact: ImpactAssessment, config: dict) -> ResourceAssignment:
    """Match appropriate responders to the incident."""
    required_skills = [classification.category.value, classification.subcategory]
    resources = check_resource_availability(required_skills, classification.urgency)
    sla = get_sla_requirements(classification.urgency, classification.category.value)

    prompt = RESOURCE_PROMPT.format(
        resources=json.dumps(resources, indent=2),
        sla=json.dumps(sla, indent=2),
        classification=classification.model_dump_json(),
        impact=impact.model_dump_json()
    )

    response = call_llm(client, provider, model, prompt,
                        config["agents"]["resource_matcher"]["temperature"],
                        config["agents"]["resource_matcher"]["max_tokens"])
    data = parse_json_response(response)

    data["primary_responder"] = Resource(**data["primary_responder"])
    data["backup_responders"] = [Resource(**r) for r in data.get("backup_responders", [])]
    return ResourceAssignment(**data)


def draft_response(client, provider: str, model: str, incident: Incident,
                   classification: Classification, impact: ImpactAssessment,
                   resources: ResourceAssignment, config: dict) -> ResponsePlan:
    """Draft communications and action plan."""
    stakeholders = get_stakeholder_list(classification.category.value, impact.impact_score, impact.blast_radius)
    templates = get_communication_templates(classification.category.value, classification.urgency)

    prompt = RESPONSE_PROMPT.format(
        stakeholders=json.dumps(stakeholders, indent=2),
        templates=json.dumps(templates, indent=2),
        incident=incident.model_dump_json(),
        classification=classification.model_dump_json(),
        impact=impact.model_dump_json(),
        resources=resources.model_dump_json()
    )

    response = call_llm(client, provider, model, prompt,
                        config["agents"]["response_drafter"]["temperature"],
                        config["agents"]["response_drafter"]["max_tokens"])
    data = parse_json_response(response)

    data["communications"] = [Communication(**c) for c in data.get("communications", [])]
    return ResponsePlan(**data)
