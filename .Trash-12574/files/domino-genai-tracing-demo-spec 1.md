# Domino GenAI Tracing Demo: Incident Triage & Response System

## Project Overview

Build a multi-agent incident triage and response system that demonstrates Domino's GenAI tracing capabilities. This demo showcases two key differentiators:

1. **Inline Evaluation** — Evaluators defined directly in the `@add_tracing` decorator
2. **Aggregated Metrics** — Declarative metric aggregation at run time using tuples

The system processes incidents (security alerts, service requests, safety reports) through a pipeline of specialized agents, each with its own tools and evaluation criteria.

---

## Technical Requirements

### Dependencies

```
domino-python-sdk (from git: https://github.com/dominodatalab/python-domino.git@master#egg=dominodatalab[data,aisystems])
mlflow==3.2.0
openai>=1.0.0
pydantic>=2.0.0
python-dotenv
pyyaml
```

### Environment Variables

```
OPENAI_API_KEY=<your-api-key>
```

### Domino SDK Imports

```python
from domino.aisystems.tracing import add_tracing, search_traces, log_evaluation
from domino.aisystems.logging import DominoRun
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Incident Input                           │
│  (ticket_id, description, source, timestamp, reporter, etc.)    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ClassifierAgent                            │
│  - Categorizes incident type (security, operational, etc.)      │
│  - Assigns urgency level (1-5)                                  │
│  - Identifies affected domain                                   │
│  Tools: lookup_category_definitions                             │
│  Inline Eval: classification_confidence, urgency_appropriate    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ImpactAssessmentAgent                        │
│  - Evaluates blast radius                                       │
│  - Estimates affected users/systems                             │
│  - Calculates financial exposure                                │
│  Tools: lookup_historical_incidents, calculate_impact_score     │
│  Inline Eval: impact_score, historical_match_quality            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ResourceMatcherAgent                         │
│  - Identifies appropriate responders                            │
│  - Checks availability and skills                               │
│  - Validates SLA requirements                                   │
│  Tools: check_resource_availability, get_sla_requirements       │
│  Inline Eval: resource_match_score, sla_compliance              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ResponseDrafterAgent                         │
│  - Generates stakeholder communication                          │
│  - Creates action plan summary                                  │
│  - Produces escalation notes if needed                          │
│  Tools: get_communication_templates, get_stakeholder_list       │
│  Inline Eval: completeness_score, tone_appropriate              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TriageResult                              │
│  (classification, impact, assigned_resources, communications)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Models

### Input Models

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class IncidentSource(str, Enum):
    MONITORING = "monitoring"
    USER_REPORT = "user_report"
    AUTOMATED_SCAN = "automated_scan"
    EXTERNAL_NOTIFICATION = "external_notification"
    AUDIT = "audit"

class Incident(BaseModel):
    """Input incident to be triaged."""
    ticket_id: str = Field(..., description="Unique identifier for the incident")
    description: str = Field(..., description="Detailed description of the incident")
    source: IncidentSource = Field(..., description="How the incident was reported")
    timestamp: datetime = Field(default_factory=datetime.now)
    reporter: Optional[str] = Field(None, description="Person or system that reported")
    affected_system: Optional[str] = Field(None, description="System mentioned in report")
    initial_severity: Optional[int] = Field(None, ge=1, le=5, description="Initial severity if provided")
```

### Output Models

```python
class IncidentCategory(str, Enum):
    SECURITY = "security"
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"
    USER_ACCESS = "user_access"

class Classification(BaseModel):
    """Output from ClassifierAgent."""
    category: IncidentCategory
    subcategory: str
    urgency: int = Field(..., ge=1, le=5, description="1=lowest, 5=critical")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    affected_domain: str

class ImpactAssessment(BaseModel):
    """Output from ImpactAssessmentAgent."""
    impact_score: float = Field(..., ge=0.0, le=10.0, description="Overall impact 0-10")
    affected_users_estimate: int
    affected_systems: list[str]
    financial_exposure: Optional[float] = Field(None, description="Estimated $ impact")
    similar_incidents: list[dict]  # Historical matches
    blast_radius: str  # "contained", "department", "organization", "external"
    reasoning: str

class Resource(BaseModel):
    """A matched resource/responder."""
    resource_id: str
    name: str
    role: str
    skills: list[str]
    availability: str  # "immediate", "within_1h", "within_4h", "next_business_day"
    match_score: float = Field(..., ge=0.0, le=1.0)

class ResourceAssignment(BaseModel):
    """Output from ResourceMatcherAgent."""
    primary_responder: Resource
    backup_responders: list[Resource]
    sla_target_hours: float
    sla_met: bool
    escalation_path: list[str]
    reasoning: str

class Communication(BaseModel):
    """A drafted communication."""
    audience: str  # "technical_team", "management", "affected_users", "external"
    subject: str
    body: str
    urgency_indicator: str

class ResponsePlan(BaseModel):
    """Output from ResponseDrafterAgent."""
    communications: list[Communication]
    action_items: list[str]
    estimated_resolution_time: str
    escalation_required: bool
    completeness_score: float = Field(..., ge=0.0, le=1.0)

class TriageResult(BaseModel):
    """Final output of the triage pipeline."""
    ticket_id: str
    classification: Classification
    impact: ImpactAssessment
    resources: ResourceAssignment
    response: ResponsePlan
    total_processing_time_seconds: float
    trace_id: Optional[str] = None
```

---

## Agent Specifications

### 1. ClassifierAgent

**Purpose:** Categorize the incident and assign urgency level.

**System Prompt:**
```
You are an incident classification specialist. Analyze the incident description and determine:
1. The primary category (security, operational, performance, data_integrity, compliance, infrastructure, user_access)
2. A specific subcategory
3. Urgency level (1-5, where 5 is critical/immediate action required)
4. The affected domain/department

Consider keywords, tone, mentioned systems, and potential business impact when classifying.

Classification Guidelines:
- Urgency 5: Active security breach, complete system outage, data loss in progress
- Urgency 4: Significant service degradation, potential security incident, compliance deadline at risk
- Urgency 3: Moderate impact, workarounds available, multiple users affected
- Urgency 2: Minor impact, single user/system affected, low business criticality
- Urgency 1: Informational, planned maintenance, minor requests
```

**Inline Evaluator:**
```python
def classifier_evaluator(inputs: dict, outputs: dict) -> dict:
    """Evaluate classification quality."""
    urgency = outputs.get("urgency", 3)
    confidence = outputs.get("confidence", 0.5)
    description = inputs.get("description", "").lower()
    
    # Check if urgency aligns with keywords
    critical_keywords = ["breach", "outage", "critical", "emergency", "down", "compromised"]
    has_critical = any(kw in description for kw in critical_keywords)
    urgency_appropriate = "appropriate" if (has_critical and urgency >= 4) or (not has_critical and urgency < 4) else "misaligned"
    
    return {
        "classification_confidence": confidence,
        "urgency_appropriate": urgency_appropriate,
        "category_assigned": outputs.get("category", "unknown")
    }
```

### 2. ImpactAssessmentAgent

**Purpose:** Evaluate the scope and severity of impact.

**System Prompt:**
```
You are an impact assessment specialist. Given an incident classification, evaluate:
1. Impact score (0-10 scale based on scope and severity)
2. Estimated number of affected users
3. List of affected systems
4. Financial exposure estimate if applicable
5. Blast radius (contained, department, organization, external)

Use historical incident data when available to inform your assessment. Consider:
- Direct vs. indirect impact
- Short-term vs. long-term consequences
- Regulatory/compliance implications
- Reputational risk
```

**Inline Evaluator:**
```python
def impact_evaluator(inputs: dict, outputs: dict) -> dict:
    """Evaluate impact assessment quality."""
    impact_score = outputs.get("impact_score", 5.0)
    urgency = inputs.get("classification", {}).get("urgency", 3)
    similar_count = len(outputs.get("similar_incidents", []))
    
    # Impact should roughly correlate with urgency
    impact_urgency_aligned = abs(impact_score / 2 - urgency) <= 1.5
    
    return {
        "impact_score": impact_score,
        "historical_matches_found": similar_count,
        "impact_urgency_alignment": "aligned" if impact_urgency_aligned else "misaligned",
        "blast_radius": outputs.get("blast_radius", "unknown")
    }
```

### 3. ResourceMatcherAgent

**Purpose:** Identify and assign appropriate responders.

**System Prompt:**
```
You are a resource allocation specialist. Based on the incident classification and impact assessment:
1. Identify the primary responder with the best skill match
2. Select 1-2 backup responders
3. Determine the SLA target based on urgency
4. Define the escalation path

Consider:
- Required skills for the incident category
- Current availability of resources
- SLA requirements based on urgency level
- Geographic/timezone considerations if relevant

SLA Guidelines by Urgency:
- Urgency 5: 1 hour response, 4 hour resolution target
- Urgency 4: 2 hour response, 8 hour resolution target
- Urgency 3: 4 hour response, 24 hour resolution target
- Urgency 2: 8 hour response, 48 hour resolution target
- Urgency 1: 24 hour response, 1 week resolution target
```

**Inline Evaluator:**
```python
def resource_evaluator(inputs: dict, outputs: dict) -> dict:
    """Evaluate resource matching quality."""
    primary = outputs.get("primary_responder", {})
    match_score = primary.get("match_score", 0.5)
    sla_met = outputs.get("sla_met", False)
    urgency = inputs.get("classification", {}).get("urgency", 3)
    availability = primary.get("availability", "next_business_day")
    
    # High urgency should have immediate availability
    availability_appropriate = (
        (urgency >= 4 and availability in ["immediate", "within_1h"]) or
        (urgency < 4)
    )
    
    return {
        "resource_match_score": match_score,
        "sla_compliance": "met" if sla_met else "at_risk",
        "availability_appropriate": "appropriate" if availability_appropriate else "delayed",
        "backup_count": len(outputs.get("backup_responders", []))
    }
```

### 4. ResponseDrafterAgent

**Purpose:** Generate communications and action plans.

**System Prompt:**
```
You are a communications specialist for incident response. Create:
1. Appropriate communications for each stakeholder group
2. Clear action items for the response team
3. Estimated resolution timeline
4. Escalation recommendations if needed

Tailor tone and detail level to audience:
- Technical team: Detailed, technical language, specific action items
- Management: Executive summary, business impact focus, timeline
- Affected users: Clear, empathetic, actionable guidance, no jargon
- External parties: Professional, measured, compliant with disclosure requirements

Ensure communications are:
- Clear and concise
- Appropriately urgent without causing panic
- Actionable where relevant
- Consistent across audiences in key facts
```

**Inline Evaluator:**
```python
def response_evaluator(inputs: dict, outputs: dict) -> dict:
    """Evaluate response quality."""
    communications = outputs.get("communications", [])
    action_items = outputs.get("action_items", [])
    completeness = outputs.get("completeness_score", 0.5)
    urgency = inputs.get("classification", {}).get("urgency", 3)
    escalation = outputs.get("escalation_required", False)
    
    # High urgency incidents should have escalation considered
    escalation_appropriate = (urgency >= 4 and escalation) or (urgency < 4)
    
    # Check communication coverage
    audiences = [c.get("audience") for c in communications]
    has_technical = "technical_team" in audiences
    has_management = "management" in audiences if urgency >= 3 else True
    
    return {
        "completeness_score": completeness,
        "communication_count": len(communications),
        "action_item_count": len(action_items),
        "escalation_appropriate": "appropriate" if escalation_appropriate else "review_needed",
        "audience_coverage": "complete" if (has_technical and has_management) else "incomplete"
    }
```

---

## Tool Specifications

### Tools for ClassifierAgent

```python
def lookup_category_definitions() -> dict:
    """
    Returns the official category definitions and classification criteria.
    Used to ensure consistent categorization.
    """
    return {
        "security": {
            "description": "Incidents involving unauthorized access, data breaches, malware, or security policy violations",
            "subcategories": ["unauthorized_access", "malware", "data_breach", "phishing", "policy_violation"],
            "keywords": ["breach", "unauthorized", "hack", "malware", "virus", "phishing", "credentials", "attack"]
        },
        "operational": {
            "description": "Incidents affecting day-to-day operations and business processes",
            "subcategories": ["process_failure", "human_error", "resource_shortage", "vendor_issue"],
            "keywords": ["failed", "error", "stuck", "delayed", "missing", "incorrect"]
        },
        "performance": {
            "description": "Incidents involving system slowness, degradation, or capacity issues",
            "subcategories": ["latency", "throughput", "capacity", "timeout"],
            "keywords": ["slow", "timeout", "latency", "degraded", "capacity", "memory", "cpu"]
        },
        "data_integrity": {
            "description": "Incidents involving data corruption, loss, or inconsistency",
            "subcategories": ["corruption", "data_loss", "sync_failure", "validation_error"],
            "keywords": ["corrupt", "lost", "missing data", "inconsistent", "mismatch", "invalid"]
        },
        "compliance": {
            "description": "Incidents with regulatory, audit, or policy compliance implications",
            "subcategories": ["regulatory", "audit_finding", "policy_breach", "certification"],
            "keywords": ["compliance", "audit", "regulation", "GDPR", "HIPAA", "SOX", "violation"]
        },
        "infrastructure": {
            "description": "Incidents affecting hardware, network, or platform infrastructure",
            "subcategories": ["hardware_failure", "network_issue", "cloud_service", "database"],
            "keywords": ["server", "network", "database", "cloud", "AWS", "Azure", "outage", "down"]
        },
        "user_access": {
            "description": "Incidents related to user authentication, authorization, or account management",
            "subcategories": ["login_issue", "permission_error", "account_locked", "sso_failure"],
            "keywords": ["login", "password", "access denied", "permission", "locked", "SSO", "MFA"]
        }
    }
```

### Tools for ImpactAssessmentAgent

```python
def lookup_historical_incidents(category: str, subcategory: str, limit: int = 5) -> list[dict]:
    """
    Search for similar historical incidents to inform impact assessment.
    Returns past incidents with their actual impact metrics.
    """
    # Simulated historical data - in production, this would query a database
    historical_db = [
        {
            "ticket_id": "INC-2024-1001",
            "category": "security",
            "subcategory": "unauthorized_access",
            "impact_score": 8.5,
            "affected_users": 150,
            "resolution_hours": 6,
            "financial_impact": 45000
        },
        {
            "ticket_id": "INC-2024-0892",
            "category": "infrastructure",
            "subcategory": "database",
            "impact_score": 7.0,
            "affected_users": 500,
            "resolution_hours": 4,
            "financial_impact": 25000
        },
        {
            "ticket_id": "INC-2024-0756",
            "category": "performance",
            "subcategory": "latency",
            "impact_score": 5.5,
            "affected_users": 1200,
            "resolution_hours": 2,
            "financial_impact": 8000
        },
        {
            "ticket_id": "INC-2024-0623",
            "category": "security",
            "subcategory": "phishing",
            "impact_score": 6.0,
            "affected_users": 25,
            "resolution_hours": 12,
            "financial_impact": 15000
        },
        {
            "ticket_id": "INC-2024-0445",
            "category": "data_integrity",
            "subcategory": "sync_failure",
            "impact_score": 7.5,
            "affected_users": 80,
            "resolution_hours": 8,
            "financial_impact": 35000
        }
    ]
    
    # Filter by category, with subcategory as secondary match
    matches = [
        inc for inc in historical_db 
        if inc["category"] == category
    ]
    
    # Prioritize subcategory matches
    matches.sort(key=lambda x: (x["subcategory"] == subcategory, x["impact_score"]), reverse=True)
    
    return matches[:limit]


def calculate_impact_score(
    urgency: int,
    affected_users: int,
    affected_systems_count: int,
    has_financial_impact: bool,
    has_compliance_implications: bool
) -> float:
    """
    Calculate a normalized impact score (0-10) based on multiple factors.
    """
    # Base score from urgency (0-4 points)
    base_score = (urgency - 1) * 1.0
    
    # User impact (0-2 points)
    if affected_users > 1000:
        user_score = 2.0
    elif affected_users > 100:
        user_score = 1.5
    elif affected_users > 10:
        user_score = 1.0
    else:
        user_score = 0.5
    
    # System impact (0-2 points)
    system_score = min(affected_systems_count * 0.5, 2.0)
    
    # Financial impact (0-1 point)
    financial_score = 1.0 if has_financial_impact else 0.0
    
    # Compliance impact (0-1 point)
    compliance_score = 1.0 if has_compliance_implications else 0.0
    
    total = base_score + user_score + system_score + financial_score + compliance_score
    return min(total, 10.0)
```

### Tools for ResourceMatcherAgent

```python
def check_resource_availability(required_skills: list[str], urgency: int) -> list[dict]:
    """
    Check available resources that match required skills.
    Returns resources sorted by match score and availability.
    """
    # Simulated resource pool - in production, this would query HR/scheduling system
    resource_pool = [
        {
            "resource_id": "RES-001",
            "name": "Alice Chen",
            "role": "Senior Security Engineer",
            "skills": ["security", "incident_response", "forensics", "network"],
            "availability": "immediate",
            "current_load": 2
        },
        {
            "resource_id": "RES-002",
            "name": "Bob Martinez",
            "role": "Infrastructure Lead",
            "skills": ["infrastructure", "database", "cloud", "networking"],
            "availability": "within_1h",
            "current_load": 3
        },
        {
            "resource_id": "RES-003",
            "name": "Carol Williams",
            "role": "DevOps Engineer",
            "skills": ["infrastructure", "performance", "monitoring", "automation"],
            "availability": "immediate",
            "current_load": 1
        },
        {
            "resource_id": "RES-004",
            "name": "David Kim",
            "role": "Data Engineer",
            "skills": ["data_integrity", "database", "ETL", "compliance"],
            "availability": "within_4h",
            "current_load": 4
        },
        {
            "resource_id": "RES-005",
            "name": "Eva Johnson",
            "role": "Compliance Officer",
            "skills": ["compliance", "audit", "policy", "security"],
            "availability": "within_1h",
            "current_load": 2
        },
        {
            "resource_id": "RES-006",
            "name": "Frank Lee",
            "role": "Support Team Lead",
            "skills": ["user_access", "troubleshooting", "communication", "escalation"],
            "availability": "immediate",
            "current_load": 3
        }
    ]
    
    def calculate_match_score(resource: dict) -> float:
        skill_overlap = len(set(resource["skills"]) & set(required_skills))
        skill_score = skill_overlap / max(len(required_skills), 1)
        
        # Availability bonus
        availability_scores = {
            "immediate": 0.2,
            "within_1h": 0.15,
            "within_4h": 0.1,
            "next_business_day": 0.0
        }
        availability_bonus = availability_scores.get(resource["availability"], 0)
        
        # Load penalty
        load_penalty = resource["current_load"] * 0.05
        
        return min(max(skill_score + availability_bonus - load_penalty, 0), 1.0)
    
    # Calculate match scores and sort
    for resource in resource_pool:
        resource["match_score"] = calculate_match_score(resource)
    
    # Filter by minimum availability for high urgency
    if urgency >= 4:
        resource_pool = [r for r in resource_pool if r["availability"] in ["immediate", "within_1h"]]
    
    return sorted(resource_pool, key=lambda x: x["match_score"], reverse=True)


def get_sla_requirements(urgency: int, category: str) -> dict:
    """
    Get SLA requirements based on urgency and incident category.
    """
    base_sla = {
        5: {"response_hours": 1, "resolution_hours": 4, "escalation_threshold_hours": 2},
        4: {"response_hours": 2, "resolution_hours": 8, "escalation_threshold_hours": 4},
        3: {"response_hours": 4, "resolution_hours": 24, "escalation_threshold_hours": 8},
        2: {"response_hours": 8, "resolution_hours": 48, "escalation_threshold_hours": 24},
        1: {"response_hours": 24, "resolution_hours": 168, "escalation_threshold_hours": 48}
    }
    
    sla = base_sla.get(urgency, base_sla[3]).copy()
    
    # Compliance and security incidents have stricter SLAs
    if category in ["compliance", "security"]:
        sla["response_hours"] = max(sla["response_hours"] * 0.5, 0.5)
        sla["escalation_threshold_hours"] = max(sla["escalation_threshold_hours"] * 0.5, 1)
    
    return sla
```

### Tools for ResponseDrafterAgent

```python
def get_communication_templates(category: str, urgency: int) -> dict:
    """
    Get communication templates appropriate for the incident type and urgency.
    """
    return {
        "technical_team": {
            "template": """
INCIDENT ALERT - {urgency_label}

Ticket: {ticket_id}
Category: {category}
Assigned To: {primary_responder}

Summary: {summary}

Immediate Actions Required:
{action_items}

Escalation Path: {escalation_path}

Please acknowledge receipt and provide status update within {response_hours} hours.
""",
            "tone": "direct and technical"
        },
        "management": {
            "template": """
INCIDENT NOTIFICATION - {urgency_label}

Executive Summary:
{executive_summary}

Business Impact:
- Affected Users: {affected_users}
- Estimated Financial Impact: {financial_impact}
- Current Status: {status}

Response:
- Lead Responder: {primary_responder}
- Target Resolution: {resolution_target}

Next Update: {next_update_time}
""",
            "tone": "professional and concise, focus on business impact"
        },
        "affected_users": {
            "template": """
Service Notice

We are aware of an issue affecting {affected_service}.

What's happening: {user_friendly_description}

What you can do: {user_actions}

Expected resolution: {resolution_estimate}

We apologize for any inconvenience and will provide updates as the situation progresses.
""",
            "tone": "empathetic and clear, avoid technical jargon"
        },
        "external": {
            "template": """
Service Status Update

{organization_name} is currently investigating an issue affecting {affected_service}.

Our team is actively working to resolve this matter. We are committed to maintaining the security and reliability of our services.

We will provide updates as more information becomes available.

For urgent inquiries, please contact: {contact_info}
""",
            "tone": "professional, measured, appropriate for public disclosure"
        }
    }


def get_stakeholder_list(category: str, impact_score: float, blast_radius: str) -> list[dict]:
    """
    Determine which stakeholders need to be notified based on incident characteristics.
    """
    stakeholders = [
        {"audience": "technical_team", "required": True, "notification_method": "immediate"}
    ]
    
    # Management notification thresholds
    if impact_score >= 5 or blast_radius in ["organization", "external"]:
        stakeholders.append({
            "audience": "management",
            "required": True,
            "notification_method": "immediate" if impact_score >= 7 else "within_1h"
        })
    
    # Affected users notification
    if blast_radius in ["department", "organization", "external"]:
        stakeholders.append({
            "audience": "affected_users",
            "required": True,
            "notification_method": "after_initial_assessment"
        })
    
    # External notification
    if blast_radius == "external" or category == "compliance":
        stakeholders.append({
            "audience": "external",
            "required": category == "compliance",  # Required for compliance, optional otherwise
            "notification_method": "after_management_approval"
        })
    
    return stakeholders
```

---

## Main Pipeline Implementation

### Entry Point: `main.py`

```python
import asyncio
from datetime import datetime
from domino.aisystems.tracing import add_tracing
from domino.aisystems.logging import DominoRun
from models import Incident, TriageResult, IncidentSource
from agents import (
    ClassifierAgent,
    ImpactAssessmentAgent,
    ResourceMatcherAgent,
    ResponseDrafterAgent
)
from config import load_config

CONFIG_PATH = "config.yaml"


@add_tracing(
    name="triage_incident",
    autolog_frameworks=["openai"],
    evaluator=lambda i, o: {
        "total_processing_seconds": o.get("total_processing_time_seconds", 0),
        "final_urgency": o.get("classification", {}).get("urgency", 0),
        "final_impact_score": o.get("impact", {}).get("impact_score", 0),
        "triage_complete": "complete" if o.get("response") else "incomplete"
    }
)
async def triage_incident(incident: Incident) -> TriageResult:
    """
    Main triage pipeline that orchestrates all agents.
    """
    start_time = datetime.now()
    
    # Initialize agents
    classifier = ClassifierAgent()
    impact_assessor = ImpactAssessmentAgent()
    resource_matcher = ResourceMatcherAgent()
    response_drafter = ResponseDrafterAgent()
    
    # Step 1: Classify the incident
    classification = await classifier.classify(incident)
    
    # Step 2: Assess impact
    impact = await impact_assessor.assess(incident, classification)
    
    # Step 3: Match resources
    resources = await resource_matcher.match(classification, impact)
    
    # Step 4: Draft response
    response = await response_drafter.draft(incident, classification, impact, resources)
    
    # Calculate total processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return TriageResult(
        ticket_id=incident.ticket_id,
        classification=classification,
        impact=impact,
        resources=resources,
        response=response,
        total_processing_time_seconds=processing_time
    )


async def run_triage_batch(incidents: list[Incident]) -> list[TriageResult]:
    """
    Process a batch of incidents with full tracing and metrics aggregation.
    """
    # Define aggregated metrics - KEY DIFFERENTIATOR
    aggregated_metrics = [
        ("classification_confidence", "mean"),
        ("classification_confidence", "stdev"),  # Shows consistency
        ("impact_score", "median"),
        ("impact_score", "max"),  # Worst case tracking
        ("resource_match_score", "mean"),
        ("total_processing_seconds", "mean"),
        ("total_processing_seconds", "max"),  # Identify slow cases
    ]
    
    results = []
    
    with DominoRun(
        agent_config_path=CONFIG_PATH,
        aggregated_metrics=aggregated_metrics
    ) as run:
        for incident in incidents:
            result = await triage_incident(incident)
            result.trace_id = run.info.run_id  # Attach trace reference
            results.append(result)
    
    return results


# Example usage
if __name__ == "__main__":
    # Sample incidents for demo
    sample_incidents = [
        Incident(
            ticket_id="INC-2024-2001",
            description="Multiple users reporting inability to access the trading platform. Error message shows 'Service Unavailable'. Started approximately 15 minutes ago. Affecting the entire APAC region.",
            source=IncidentSource.USER_REPORT,
            reporter="Help Desk",
            affected_system="Trading Platform",
            initial_severity=4
        ),
        Incident(
            ticket_id="INC-2024-2002",
            description="Automated security scan detected unusual outbound traffic patterns from server DB-PROD-03. Traffic destination is an unrecognized external IP. No known maintenance or data sync scheduled.",
            source=IncidentSource.AUTOMATED_SCAN,
            affected_system="DB-PROD-03",
            initial_severity=5
        ),
        Incident(
            ticket_id="INC-2024-2003",
            description="Monthly compliance report generation failed. Error in data aggregation step. Report is due to regulators in 72 hours.",
            source=IncidentSource.MONITORING,
            affected_system="Compliance Reporting System",
            initial_severity=3
        ),
        Incident(
            ticket_id="INC-2024-2004",
            description="User john.smith@company.com unable to reset password via self-service portal. MFA token not being accepted. User has critical presentation in 2 hours.",
            source=IncidentSource.USER_REPORT,
            reporter="john.smith@company.com",
            affected_system="Identity Management",
            initial_severity=2
        )
    ]
    
    # Run the triage batch
    results = asyncio.run(run_triage_batch(sample_incidents))
    
    # Output results
    for result in results:
        print(f"\n{'='*60}")
        print(f"Ticket: {result.ticket_id}")
        print(f"Category: {result.classification.category} / {result.classification.subcategory}")
        print(f"Urgency: {result.classification.urgency}")
        print(f"Impact Score: {result.impact.impact_score}")
        print(f"Primary Responder: {result.resources.primary_responder.name}")
        print(f"SLA Met: {result.resources.sla_met}")
        print(f"Processing Time: {result.total_processing_time_seconds:.2f}s")
```

---

## Configuration File

### `config.yaml`

```yaml
# Model configurations
models:
  default: "gpt-4o-mini"
  classifier: "gpt-4o-mini"
  impact_assessor: "gpt-4o-mini"
  resource_matcher: "gpt-4o-mini"
  response_drafter: "gpt-4o-mini"

# Agent configurations
agents:
  classifier:
    temperature: 0.3
    max_tokens: 500
  impact_assessor:
    temperature: 0.4
    max_tokens: 800
  resource_matcher:
    temperature: 0.2
    max_tokens: 600
  response_drafter:
    temperature: 0.7
    max_tokens: 1500

# System prompts (abbreviated - full prompts in agent files)
prompts:
  classifier_system: |
    You are an incident classification specialist...
  impact_assessor_system: |
    You are an impact assessment specialist...
  resource_matcher_system: |
    You are a resource allocation specialist...
  response_drafter_system: |
    You are a communications specialist...

# SLA configurations
sla:
  urgency_5:
    response_hours: 1
    resolution_hours: 4
  urgency_4:
    response_hours: 2
    resolution_hours: 8
  urgency_3:
    response_hours: 4
    resolution_hours: 24
  urgency_2:
    response_hours: 8
    resolution_hours: 48
  urgency_1:
    response_hours: 24
    resolution_hours: 168

# Evaluation thresholds
evaluation:
  min_confidence_threshold: 0.7
  impact_urgency_tolerance: 1.5
  min_resource_match_score: 0.6
```

---

## Project Structure

```
domino-incident-triage-demo/
├── README.md                    # Project overview and setup instructions
├── requirements.txt             # Python dependencies
├── config.yaml                  # Centralized configuration
├── main.py                      # Entry point and batch processing
├── models/
│   ├── __init__.py
│   ├── inputs.py               # Incident input models
│   └── outputs.py              # Output models (Classification, Impact, etc.)
├── agents/
│   ├── __init__.py
│   ├── base.py                 # Base agent class
│   ├── classifier.py           # ClassifierAgent with inline eval
│   ├── impact_assessor.py      # ImpactAssessmentAgent with inline eval
│   ├── resource_matcher.py     # ResourceMatcherAgent with inline eval
│   └── response_drafter.py     # ResponseDrafterAgent with inline eval
├── tools/
│   ├── __init__.py
│   ├── classification.py       # lookup_category_definitions
│   ├── impact.py               # lookup_historical_incidents, calculate_impact_score
│   ├── resources.py            # check_resource_availability, get_sla_requirements
│   └── communications.py       # get_communication_templates, get_stakeholder_list
├── data/
│   ├── sample_incidents.csv    # Sample incident data for demo
│   ├── historical_incidents.json  # Simulated historical data
│   └── resource_pool.json      # Simulated resource pool
├── notebooks/
│   └── demo_walkthrough.ipynb  # Interactive demo notebook
└── tests/
    ├── test_agents.py
    ├── test_tools.py
    └── test_pipeline.py
```

---

## Key Implementation Notes

### Demonstrating Inline Evaluation (Differentiator #1)

Every agent should have the `evaluator` parameter in its `@add_tracing` decorator:

```python
@add_tracing(
    name="classify_incident",
    autolog_frameworks=["openai"],
    evaluator=classifier_evaluator  # Function defined in same module
)
async def classify(self, incident: Incident) -> Classification:
    ...
```

The evaluator function receives `inputs` and `outputs` dicts and returns a dict of metrics/labels.

### Demonstrating Aggregated Metrics (Differentiator #2)

The `DominoRun` context manager should always include `aggregated_metrics`:

```python
with DominoRun(
    agent_config_path=CONFIG_PATH,
    aggregated_metrics=[
        ("classification_confidence", "mean"),
        ("classification_confidence", "stdev"),
        ("impact_score", "median"),
        ("resource_match_score", "mean"),
        ("total_processing_seconds", "mean"),
    ]
) as run:
    # Process incidents
```

This automatically computes aggregated statistics across all traces in the run.

---

## Sample Incidents for Demo

Include diverse incidents to show cross-vertical appeal:

| Ticket ID | Vertical Appeal | Incident Type |
|-----------|-----------------|---------------|
| INC-2024-2001 | Financial Services | Trading platform outage |
| INC-2024-2002 | All (Security) | Suspicious network traffic |
| INC-2024-2003 | Financial/Public Sector | Compliance reporting failure |
| INC-2024-2004 | All (IT Support) | User access issue |
| INC-2024-2005 | Healthcare | Patient data sync failure |
| INC-2024-2006 | Energy/Utilities | SCADA system anomaly |
| INC-2024-2007 | Public Sector | Citizen portal degradation |
| INC-2024-2008 | All (Data) | Database replication lag |

---

## Demo Talking Points

When presenting the demo, emphasize:

1. **Inline Evaluation Simplicity**
   - "Notice how evaluation is defined right in the decorator - no separate evaluation pipeline needed"
   - "Each agent evaluates its own output quality in real-time"
   - "Compare this to competitors who require separate evaluation steps"

2. **Aggregated Metrics**
   - "At the end of a batch run, we automatically get statistical summaries"
   - "Mean confidence tells us overall classifier quality"
   - "Stdev shows consistency - are some incident types harder to classify?"
   - "Max processing time helps identify edge cases that need optimization"

3. **Configuration-Driven (Future Differentiator)**
   - "All configuration lives in one YAML file"
   - "In the next release, you'll be able to deploy this config directly as an agent"
   - "No serialization or wrapping required"

4. **Cross-Vertical Applicability**
   - "The same triage pattern applies to any organization"
   - "Swap in your incident categories and resource pools"
   - "The evaluation criteria remain consistent"

---

## Testing Checklist

Before demoing, verify:

- [ ] All agents trace correctly in Domino Experiment Manager
- [ ] Inline evaluators produce expected metrics for each agent
- [ ] Aggregated metrics appear in the run summary
- [ ] Historical incident lookup returns relevant matches
- [ ] Resource matching considers urgency appropriately
- [ ] Response drafting produces audience-appropriate communications
- [ ] End-to-end pipeline completes for all sample incidents
- [ ] Trace comparison view works for comparing runs
