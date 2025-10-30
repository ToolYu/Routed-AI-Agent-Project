from __future__ import annotations  # Allow forward references in type annotations

# ====== Standard Library ======
import csv
import json
import os
import time
from dataclasses import dataclass, field  # Concise data class definitions
from pathlib import Path                  # Filesystem path utilities
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict  # Typing helpers

# ====== Third‑Party Libraries ======
# NumPy 2.x compatibility shim for libraries that still reference `numpy.rec`
try:  # lightweight and safe: set alias proactively without hasattr/getattr checks
    import numpy as _np  # type: ignore
    import numpy.core.records as _records  # type: ignore
    # Expose `np.rec` and module path `numpy.rec`
    setattr(_np, "rec", _records)
    import sys as _sys  # local import to avoid leaking globally
    _sys.modules.setdefault("numpy.rec", _records)
except Exception:
    pass

import pandas as pd                       # CSV reading and data wrangling
from tqdm import tqdm                     # Progress bar
from dotenv import find_dotenv, load_dotenv  # Load environment variables from .env

# LangChain core components
from langchain_core.language_models.chat_models import BaseChatModel  # Base class for chat models
from langchain_core.output_parsers import JsonOutputParser            # Parse model output into structured JSON
from langchain_core.prompts import ChatPromptTemplate                 # Prompt template builder

# LangGraph for multi‑agent orchestration
from langgraph.graph import END, StateGraph  # END marks terminal node; StateGraph defines the DAG

# Pydantic for data modeling and validation
from pydantic import BaseModel, Field

# ====== Optional Imports: LLM Providers ======
try:
    from langchain_groq import ChatGroq  # Groq (very fast inference)
except Exception:
    ChatGroq = None  # Fallback to None if the package is not installed

try:
    from langchain_openai import ChatOpenAI  # OpenAI models (GPT, etc.)
except Exception:
    ChatOpenAI = None

try:
    from langchain_openai import AzureChatOpenAI  # Azure‑hosted OpenAI
except Exception:
    AzureChatOpenAI = None

# =============================
# ConstraintConfig: validation/guardrail settings
# =============================
@dataclass
class ConstraintConfig:
    """Guardrail configuration controlling the constraint checks."""

    linkedin_char_limit: int = 300       # Max characters for LinkedIn copy
    email_subject_limit: int = 70        # Max subject length
    email_word_range: tuple[int, int] = (100, 300)  # Target email body word range

    # Common boilerplate phrases to avoid
    banned_phrases: Iterable[str] = field(
        default_factory=lambda: (
            "I hope this message finds you well",
            "I hope you're doing well",
            "I am reaching out to connect",
            "I came across your profile",
            "I wanted to follow up",
            "I wanted to reach out",
            "As an AI language model",
            "As a highly skilled professional",
            "Hi there",
            "quick hello",
            "swap notes",
            "fellow practitioner",
            "To whom it may concern",
            "Allow me to introduce myself",
            "I am writing to you",
            "In today’s fast-paced world",
            "Furthermore",
            "Additionally",
            "I appreciate your prompt reply",
            "If you have any questions, please let me know",
            "Thank you for your time and consideration",
            "I look forward to hearing from you",
            "Please do not hesitate to contact me",
            "Best regards",
            "Sincerely yours",
            "With kind regards",
            "I would like to inquire about",
            "Per our last conversation",
            "At your earliest convenience",
            "It is worth noting that",
            "I wanted to follow up",
            "As mentioned earlier",
            "It is important to note",
            "This email is to inform you",
            "Please find attached",
            "Kindly note",
            "Needless to say",
            "Out-of-the-box",
            "Leverage",
            "Seamless integration",
            "Cutting-edge",
            "Game changer",
            "Next-gen",
            "Paradigm shift",
            "Holistic approach",
            "Synergy",
            "Optimize",
            "Maximize",
            "Core competency",
            "Move the needle",
            "Out-of-the-box thinking",
            "Take it to the next level",
            "Drill down",
            "Circle back",
            "Deep dive",
            "Pivot",
            "Agile",
            "Scalable",
            "Real-time",
            "Best-in-class",
            "Empower",
            "Enhance",
            "Unlock the potential",
            "Future-proof",
            "Innovative solutions",
            "Transformative",
            "Disruptive technology",
            "Exciting opportunities",
            "Remarkable success",
            "Outstanding",
            "Inspiring",
            "I’ve come across",
            "It’s obvious that",
            "Please let me know if you have questions",
            "I look forward to your response",
            "I’m eager to know",
            "Thank you for your consideration",
            "I hope to discuss this further",
            "Please don't hesitate to reach out",
            "Kindly",
            "I would be happy to",
            "At your convenience",
            "Attached is",
            "Please find attached"
        )
    )


    min_personal_tokens: int = 3  # At least 3 concrete target‑specific references
    max_iterations: int = 3       # Upper bound for refinement loops

    def __post_init__(self) -> None:
        """Freeze banned_phrases as a tuple for deterministic ordering."""
        self.banned_phrases = tuple(self.banned_phrases)

class PersonaBriefSchema(BaseModel):
    """Structured summary that anchors message personalization."""

    persona_summary: str = Field(..., description="One-paragraph synthesis of the target and outreach hook.")
    overlaps: List[str] = Field(
        default_factory=list,
        description="Bullet list of concrete shared experiences, interests, or credentials.",
    )
    motivation: str = Field(..., description="Why the outreach matters for the sender right now.")
    call_to_action: str = Field(..., description="Specific ask that sets a next step or clear intent.")
    tone: List[str] = Field(
        default_factory=list,
        description="Voice and tone guardrails (e.g., warm, succinct, collegial).",
    )
    personalization_angle: str = Field(
        ...,
        description="Narrative thread that ties the message to the target's unique context.",
    )
    # Extracted factual slots and constraint tags (optional)
    slot_map: Dict[str, Any] = Field(default_factory=dict, description="Extracted factual fields and allowed/missing keys")
    tone_tags: List[str] = Field(default_factory=list, description="Subset of {succinct,warm,collegial,confident}")
    min_personal: int = Field(default=3, description="Target count of concrete references to include")


class OutreachMessageSchema(BaseModel):
    """Single message tailored for one delivery channel."""

    channel: str = Field(..., description="Delivery channel such as email, linkedin, sms.")
    subject: Optional[str] = Field(
        None, description="Subject line when the channel requires one. Omit for channels that do not use subjects."
    )
    opening: str = Field(..., description="Personalized salutation and hook.")
    body: str = Field(..., description="Main content with highlights and narrative.")
    closing: str = Field(..., description="CTA tied to the motivation plus reciprocity or compliment.")
    signature: str = Field(..., description="Human signoff that matches the requested tone.")


class ContentAgentOutputSchema(BaseModel):
    """Collection of channel-specific drafts."""

    rationale: str = Field(..., description="Summary of the personalization strategy that was applied.")
    messages: List[OutreachMessageSchema] = Field(
        ..., description="Drafts per channel, already respecting length and tone guidance."
    )


class ConstraintIssueSchema(BaseModel):
    """Single issue flagged by the constraint agent."""

    category: str = Field(..., description="One of tone, personalization, length, or format.")
    severity: str = Field(..., description="Severity bucket such as high, medium, low.")
    detail: str = Field(..., description="What went wrong with evidence from the draft.")
    fix: str = Field(..., description="Actionable adjustment to resolve the issue.")


class ConstraintReportSchema(BaseModel):
    """Structured evaluation output from the constraint agent."""

    is_satisfied: bool = Field(..., description="True when the copy is ready to send.")
    score: int = Field(..., description="Holistic score from 1-10 where >=8 implies high confidence.")
    summary: str = Field(..., description="Narrative evaluation that references the brief and target context.")
    issues: List[ConstraintIssueSchema] = Field(default_factory=list, description="List of specific problems.")
    refined_guidance: str = Field(
        default="",
        description="Updated creative direction that content agent should follow when re-drafting.",
    )


class OutreachState(TypedDict, total=False):
    """Shared state across the LangGraph workflow."""

    # Inputs
    user_profile: Dict[str, Any]
    target_record: Dict[str, Any]
    recommended_channels: List[str]
    selected_channels: List[str]
    intent_to_connect: str

    # Intermediate artifacts
    persona_brief: Dict[str, Any]
    draft_bundle: Dict[str, Any]
    stored_plan: Dict[str, Any]
    constraint_report: Dict[str, Any]

    # Flow control
    pending_refinement: bool
    iterations: int
    max_iterations: int
    progress_cb: Any


def _persona_agent(llm: BaseChatModel) -> JsonOutputParser:
    """Creates the runnable for the PersonaAgent."""

    persona_parser = JsonOutputParser(pydantic_object=PersonaBriefSchema)
    persona_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    '''You are PersonaAgent, a strategist who distills a human brief for personalized outreach.
Extract only verifiable overlaps and resist filler phrases. Capture tone preferences that feel organic.
Return a JSON that includes both a narrative brief AND a slot map of factual fields extracted from the sender/target.
Schema extensions (in addition to the brief fields):
- slot_map: {{
  "recipient_name": str | null,
  "recipient_company": str | null,
  "recipient_title": str | null,
  "recipient_school": str | null,
  "recipient_location": str | null,
  "recipient_projects": [str],
  "allowed_keys": ["company","title","school","location","project","skills","recent_post","achievement"],
  "missing_keys": [str]
}}
- tone_tags: from {{succinct,warm,collegial,confident}}
- min_personal: integer (>=3)
Only include verifiable facts present in profiles. Never invent.
{format_instructions}'''
                ),
            ),
            (
                "human",
                (
                    "Sender profile (YAML):\n```yaml\n{user_profile}\n```\n"
                    "Target profile (YAML):\n```yaml\n{target_record}\n```\n"
                    "Outreach intent/purpose: {intent_to_connect}\n"
                    "Extract and normalize names and entities for slot_map; if a field is absent, add it to missing_keys.\n"
                    "Focus on shared motivations, resonant accomplishments, and an approachable CTA that explicitly reflects the intent."
                ),
            ),
        ]
    )

    # Wire prompt → LLM → parser into a runnable chain
    return persona_prompt.partial(format_instructions=persona_parser.get_format_instructions()) | llm | persona_parser

def _content_agent(llm: BaseChatModel) -> JsonOutputParser:
    """Creates the runnable for the ContentAgent."""

    content_parser = JsonOutputParser(pydantic_object=ContentAgentOutputSchema)
    content_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are ContentAgent. Convert the persona brief into channel-ready copy.\n"
                    "Each draft must feel human: varied sentence lengths, concrete detail, and a natural voice aligned with tone.\n"
                    "Avoid clichés and template structures. Show specific knowledge of the target (facts from slot_map).\n"
                    "For 'email', follow this framework: 1) Subject 2) Greeting (use real name) 3) Opening (who/why) 4) Value (2–3 specific outcomes) 5) CTA (aligned to intent) 6) Closing + signature.\n"
                    "Length: aim 140–220 words (max 260). Do not remove concrete details just to shorten.\n"
                    "Hard constraints:\n"
                    "- Use recipient_name in greeting if available; do NOT use 'Hi there' when a name exists.\n"
                    "- Subject must include at least one of {{company|project|role}} and align with outreach intent.\n"
                    "- Banned (case-insensitive): 'I came across your profile', 'I am reaching out to connect', 'I hope this message finds you well', 'quick hello', 'swap notes', 'fellow practitioner'.\n"
                    "- No placeholders like '[Your Name]' or '{{name}}'. Signature must use sender's actual name from user_profile.\n"
                    "- Use at least {min_personal} concrete references from slot_map.allowed_keys; never invent missing keys.\n"
                    "- Caps: email.subject<=70 chars; email.body 100–180 words; linkedin total<=300 chars.\n"
                    "Quality process: Internally PLAN (hook + 2–3 target facts + CTA + channel constraints), generate 2–3 variants, pick best by tone/personalization/length, SELF-CHECK, then return ONLY JSON.\n"
                    "Return ONLY valid JSON matching the schema — no extra prose.\n"
                    "{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "Persona brief (JSON):\n```json\n{persona_brief}\n```\n"
                    "Sender profile (YAML):\n```yaml\n{user_profile}\n```\n"
                    "Target profile (YAML):\n```yaml\n{target_record}\n```\n"
                    "Channels to cover: {selected_channels}\n"
                    "Outreach intent/purpose: {intent_to_connect}\n"
                    "Latest refinement guidance: {refinement_notes}\n"
                    "From Persona brief, use slot_map strictly: {{recipient_name, recipient_company, recipient_title, recipient_school, recipient_location, recipient_projects, allowed_keys, missing_keys}}.\n"
                    "Respect channel formatting norms and keep copy concise.\n"
                    "Important: If 'email' is among selected channels, include exactly one email and use the recipient's actual name and facts from the target profile."
                ),
            ),
        ]
    )

    # Return runnable chain (Prompt → LLM → JSON parser)
    return (
        content_prompt.partial(
            format_instructions=content_parser.get_format_instructions(),
            min_personal=3,
        )
        | llm
        | content_parser
    )

def _constraint_agent(llm: BaseChatModel, cfg: ConstraintConfig) -> JsonOutputParser:
    """Creates the runnable for the ConstraintAgent."""

    constraint_parser = JsonOutputParser(pydantic_object=ConstraintReportSchema)
    banned_list = [str(phrase) for phrase in cfg.banned_phrases]  # Extract banned phrases from config
    banned = ", ".join(f'"{phrase}"' for phrase in banned_list)  # Join into a prompt-friendly string
    constraint_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are ConstraintAgent. Act as an executable copy auditor — verify constraints and produce actionable fixes.\n"
                    "Hard caps and checks:\n"
                    f"- LinkedIn total <= {cfg.linkedin_char_limit} chars; Email subject <= {cfg.email_subject_limit} chars; Email body {cfg.email_word_range[0]}–{cfg.email_word_range[1]} words.\n"
                    "- Greeting must use recipient_name if available; if name exists but greeting uses 'Hi there', fail with fix: Replace with 'Hi {{recipient_name}},'.\n"
                    "- Subject must include at least one of {{company|project|role}} and align with intent.\n"
                    f"- Banned phrases (case-insensitive): {banned}.\n"
                    "- No placeholders like '[Your Name]' or '{{name}}'. Signature must use sender's actual name.\n"
                    "- Personalization: Count concrete references drawn from persona.slot_map.allowed_keys across opening/body/closing; require >= {min_personal}. If less, list which keys (e.g., company, project) can be added.\n"
                    "- Opening first sentence must contain a concrete fact about the target (company/project/school/location).\n"
                    "Output a JSON report with: is_satisfied, score (1–10), summary, issues[{{category,severity,detail,fix}}], refined_guidance (bullet list fixes ContentAgent can follow).\n"
                    "If constraints fail, be specific and propose minimal, concrete edits (do not over-shorten).\n"
                    "{format_instructions}"
                ),
            ),
            (
                "human",
                (
                    "Persona brief:\n```json\n{persona_brief}\n```\n"
                    "Draft bundle:\n```json\n{draft_bundle}\n```\n"
                    "Metrics: {metrics}\n"
                    "Respond with a JSON evaluation."
                ),
            ),
        ]
    )

    # Return the runnable chain: Prompt → LLM → JSON parser
    return (
        constraint_prompt.partial(
            format_instructions=constraint_parser.get_format_instructions(),
            min_personal=max(3, cfg.min_personal_tokens),
        )
        | llm
        | constraint_parser
    )


def _compute_metrics(drafts: List[Dict[str, Any]], cfg: ConstraintConfig) -> Dict[str, Any]:
    """Collect quick stats to support constraint decisions."""

    metrics: Dict[str, Any] = {"per_channel": []}  # Aggregate by channel
    banned_phrases = [str(phrase) for phrase in cfg.banned_phrases]
    for draft in drafts:
        channel = draft.get("channel", "unknown")
        subject = draft.get("subject") or ""
        body = draft.get("body", "")
        opening = draft.get("opening", "")
        closing = draft.get("closing", "")

        body_words = len(body.split())
        linkedin_chars = len(" ".join([opening, body, closing]))  # Rough LI character count (opening+body+closing)
        contains_banned = []

        for text in (opening, body, closing, subject):
            lower_text = text.lower()
            contains_banned.extend(
                phrase
                for phrase in banned_phrases
                if phrase.lower() in lower_text
            )

        metrics["per_channel"].append(
            {
                "channel": channel,
                "word_count": body_words,
                "char_count": linkedin_chars,
                "subject_length": len(subject),
                "detected_banned_phrases": list(set(contains_banned)),
            }
        )
    return metrics

def _channel_choice(state: OutreachState) -> OutreachState:
    """Resolve the final channel plan the content agent should cover."""

    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("Channel Choice")  # Indicate the current stage

    selected = state.get("selected_channels") or state.get("recommended_channels") or []
    if not selected:
        # Default to email when no recommendation is supplied.
        selected = ["email"]

    new_state = dict(state)
    new_state["selected_channels"] = selected
    return new_state

def _persona_node(persona_chain: JsonOutputParser, state: OutreachState) -> OutreachState:
    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("PersonaAgent")

    persona = persona_chain.invoke(
        {
            "user_profile": state.get("user_profile", {}),
            "target_record": state.get("target_record", {}),
            "intent_to_connect": state.get("intent_to_connect", "networking purposes"),
        }
    )
    updated = dict(state)
    updated["persona_brief"] = persona
    return updated

def _content_node(content_chain: JsonOutputParser, state: OutreachState) -> OutreachState:
    """Debug version: show model output and prevent invalid 'purpose'-only content."""

    cb = state.get("progress_cb")
    if callable(cb):
        cb("ContentAgent")

    refinement_notes = ""
    report = state.get("constraint_report")
    if report and not report.get("is_satisfied", True):
        refinement_notes = report.get("refined_guidance") or ""

    # Execute ContentAgent
    try:
        content = content_chain.invoke(
            {
                "persona_brief": json.dumps(state.get("persona_brief", {}), ensure_ascii=False, indent=2),
                "user_profile": state.get("user_profile", {}),
                "target_record": state.get("target_record", {}),
                "selected_channels": state.get("selected_channels", []),
                "intent_to_connect": state.get("intent_to_connect", "networking purposes"),
                "refinement_notes": refinement_notes or "Stay personal, grounded, and conversational.",
            }
        )
    except Exception as e:
        print(f"[Error in ContentAgent invoke]: {e}")
        content = {}

    # Print raw content for debugging
    print("\n==================== DEBUG: RAW CONTENT OUTPUT ====================")
    print(content)
    print("===================================================================\n")

    # Minimal post-processing to fix placeholders and fill missing fields
    content = _postprocess_and_personalize_content(content, state)

    # Validate structure (must include at least one message)
    valid = (
        isinstance(content, dict)
        and "messages" in content
        and isinstance(content["messages"], list)
        and len(content["messages"]) > 0
    )

    if not valid:
        print("[Warning] Invalid ContentAgent output detected. Replacing with fallback draft...")
        # Build a simple placeholder draft to keep the pipeline stable
        fallback_message = {
            "channel": "email",
            "subject": "Hello!",
            "opening": "Hi there, I wanted to connect with you briefly.",
            "body": "It seems the model returned an incomplete message. This is a fallback template for debugging.",
            "closing": "Looking forward to connecting soon!",
            "signature": "Best,\nQianyu",
        }
        content = {
            "rationale": "Fallback message used due to invalid model output.",
            "messages": [fallback_message],
        }

    # Update state
    updated = dict(state)
    updated["draft_bundle"] = content
    return updated

def _store_plan_node(state: OutreachState) -> OutreachState:
    """Cache the latest draft bundle and brief for downstream steps."""

    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("StorePlan")

    updated = dict(state)
    updated["stored_plan"] = {
        "persona_brief": state.get("persona_brief"),
        "draft_bundle": state.get("draft_bundle"),
    }
    return updated

def _constraint_node(
    constraint_chain: JsonOutputParser, cfg: ConstraintConfig, state: OutreachState
) -> OutreachState:
    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("ConstraintAgent")

    drafts = state.get("draft_bundle", {}).get("messages", [])
    metrics = _compute_metrics(drafts, cfg)
    report = constraint_chain.invoke(
        {
            "persona_brief": json.dumps(state.get("persona_brief", {}), ensure_ascii=False, indent=2),
            "draft_bundle": json.dumps(state.get("draft_bundle", {}), ensure_ascii=False, indent=2),
            "metrics": metrics,
        }
    )

    updated = dict(state)
    updated["constraint_report"] = report
    updated["pending_refinement"] = not report.get("is_satisfied", False)
    updated.setdefault("iterations", 0)
    updated.setdefault("max_iterations", cfg.max_iterations)
    return updated

def _refinement_node(state: OutreachState) -> OutreachState:
    """Prepare the next iteration when the draft is blocked by constraints."""

    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("Refinement")

    report = state.get("constraint_report", {})  # Current report
    issues = report.get("issues", [])  # Issues list
    guidance = report.get("refined_guidance", "")  # Existing guidance
    synthesized_guidance = guidance  # Initial combined guidance

    if not synthesized_guidance:  # Build guidance from issues when missing
        bullet_list = "\n".join(
            f"- ({issue.get('category', 'general')}) {issue.get('fix', issue.get('detail', ''))}"
            for issue in issues
        )
        synthesized_guidance = (
            "Tighten voice and integrate the following fixes:\n"
            f"{bullet_list}\n"
            "Maintain warmth and specificity."
        )

    updated = dict(state)
    updated["constraint_report"]["refined_guidance"] = synthesized_guidance
    updated["iterations"] = updated.get("iterations", 0) + 1
    updated["pending_refinement"] = False
    return updated

def _route_constraint(state: OutreachState) -> str:
    """Determine next step after constraint evaluation."""

    if not state.get("pending_refinement"):
        return "confirm"

    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 2)
    if iterations >= max_iterations:
        return "force_confirm"
    return "refine"

def _confirm_choice_node(state: OutreachState) -> OutreachState:
    """Mark the workflow as complete and surface the final drafts."""

    cb = state.get("progress_cb")  # Progress callback
    if callable(cb):
        cb("Confirm Choice")

    report = state.get("constraint_report", {})
    status = "approved" if not state.get("pending_refinement") else "force_approved"
    updated = dict(state)
    updated["final_status"] = status
    updated["final_messages"] = state.get("draft_bundle", {}).get("messages", [])
    updated["evaluation"] = report
    return updated

def build_outreach_graph(
    llm: BaseChatModel,
    constraint_config: Optional[ConstraintConfig] = None,
) -> StateGraph:
    """
    Build the end-to-end LangGraph workflow.

    Parameters
    ----------
    llm:
        LangChain-compatible chat model (e.g., ChatGroq, ChatOpenAI).
    constraint_config:
        Optional overrides for constraint thresholds.
    """

    cfg = constraint_config or ConstraintConfig()
    persona_chain = _persona_agent(llm)
    content_chain = _content_agent(llm)

    workflow = StateGraph(OutreachState)

    # Nodes (each corresponds to a stage)
    workflow.add_node("channel_choice", _channel_choice)
    workflow.add_node("persona_agent", lambda state: _persona_node(persona_chain, state))
    workflow.add_node("content_agent", lambda state: _content_node(content_chain, state))
    workflow.add_node("store_plan", _store_plan_node)
    # No refinement loop for now → go straight to confirm
    workflow.add_node("confirm_choice", _confirm_choice_node)

    # Entry point
    workflow.set_entry_point("channel_choice")

    # Edges
    workflow.add_edge("channel_choice", "persona_agent")
    workflow.add_edge("persona_agent", "content_agent")
    workflow.add_edge("content_agent", "store_plan")
    # No ConstraintAgent: go to confirm after storing
    workflow.add_edge("store_plan", "confirm_choice")

    # Final node
    workflow.add_edge("confirm_choice", END)

    return workflow

def run_outreach_workflow(
    graph: StateGraph,  # Compiled LangGraph workflow
    user_profile: Dict[str, Any],  # Sender profile
    target_record: Dict[str, Any],  # Target profile
    recommended_channels: Optional[List[str]] = None,  # Suggested channels (e.g., email, linkedin)
    channel_override: Optional[List[str]] = None,  # Optional: explicitly choose channels
    max_iterations: Optional[int] = None,  # Optional: cap refinement rounds
    progress_callback: Optional[Any] = None,  # Optional: progress hook per stage
    intent_to_connect: Optional[str] = None,  # Optional: explicit outreach intent
) -> Dict[str, Any]:
    """
    Convenience helper to execute the compiled LangGraph workflow.

    Returns the final state including persona brief, drafts, and evaluation report.
    """

    starting_state: OutreachState = {  # Initial state
        "user_profile": user_profile,
        "target_record": target_record,
        "recommended_channels": recommended_channels or [],
    }

    if channel_override:
        starting_state["selected_channels"] = channel_override

    if max_iterations is not None:
        starting_state["max_iterations"] = max_iterations

    if progress_callback is not None:
        starting_state["progress_cb"] = progress_callback

    if intent_to_connect is not None:
        starting_state["intent_to_connect"] = intent_to_connect

    compiled = graph if hasattr(graph, "invoke") else graph.compile()
    # Use the provided compiled graph if available; otherwise compile it.

    return compiled.invoke(starting_state)
    # Execute the workflow from the starting state until END

__all__ = [
    "ConstraintConfig",
    "build_outreach_graph",
    "run_outreach_workflow",
]

# ---------------------------
# Convenience: sample runner
# ---------------------------

SAMPLE_DATA_PATH = "user_features.csv"       # Sample data path
OUTPUT_CSV_PATH = "personalized_messages.csv" # Output CSV path
INTENT_TO_CONNECT = "networking purposes"     # Default outreach intent

def intent_input_widget(default_intent: str = INTENT_TO_CONNECT):
    """Return an ipywidgets Textarea for entering outreach intent (for notebooks).

    Usage (in a notebook):
        from langgraph_outreach_pipeline import intent_input_widget, run_outreach_workflow, build_outreach_graph
        w = intent_input_widget("I’d appreciate a referral for the Data Scientist role.")
        # ... edit the text box ...
        intent = w.value
        state = run_outreach_workflow(graph, user_profile, target_record, recommended_channels=["email"], intent_to_connect=intent)
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ipywidgets is required for the intent input widget. Install with `pip install ipywidgets`."
        ) from exc

    ta = widgets.Textarea(
        value=str(default_intent or ""),
        description="Intent",
        placeholder="Type your outreach intent...",
        layout=widgets.Layout(width="100%", height="80px"),
    )
    display(ta)
    return ta
VERSION_TAG = "langgraph_v1"                  # Version tag

def _load_sample_profiles(path: str = SAMPLE_DATA_PATH) -> Tuple[int, List[int], Dict[int, Dict[str, Any]]]:
    df = pd.read_csv(path).fillna("")  # Read CSV and fill blanks
    if len(df) < 2:  # Require at least 2 rows (1 sender + >=1 target)
        raise ValueError("Sample dataset must contain at least 6 rows.")
    df["user_id"] = df["user_id"].astype(int)  # Ensure user_id is int
    primary_user_row = df.iloc[0]  # First row is the primary user
    target_rows = df.iloc[1:2]     # Subsequent rows are targets (sample subset)
    primary_user_id = int(primary_user_row["user_id"])
    target_ids = [int(uid) for uid in target_rows["user_id"].tolist()]  # Extract target IDs
    profile_map: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iloc[:2].iterrows():  # Build a small id→profile map
        profile_map[int(row["user_id"])] = row.to_dict()
    return primary_user_id, target_ids, profile_map

def _prepare_sample_context() -> Tuple[int, str, List[int], Dict[int, Dict[str, Any]]]:
    ru_id, targets, profiles = _load_sample_profiles()  # Load sample data
    for uid, prof in profiles.items():  # Add USER_ID to each profile
        prof["USER_ID"] = uid
    return ru_id, str(ru_id), targets, profiles

def _compose_text_from_draft(draft: Dict[str, Any]) -> str:
    """Convert a structured draft into plain text with clean line breaks.

    Rules:
    - Subject on the first line as `Subject: ...` (extracted later when sending)
    - Greeting/opening on its own line, then a blank line
    - Body paragraphs as-is
    - One blank line, then closing, then a single newline, then signature
    """

    channel = (draft.get("channel") or "").lower()
    subject = (draft.get("subject") or "").strip()
    opening = (draft.get("opening") or "").strip()
    body = (draft.get("body") or "").strip()
    closing = (draft.get("closing") or "").strip()
    signature = (draft.get("signature") or "").strip()

    # Ensure greeting line is only "Dear XXX," with a line break afterwards.
    # If opening is like "Dear Name, <extra>", split at the first comma/， and
    # prepend the remainder to the body as the first paragraph.
    import re as _re
    m = _re.match(r"^\s*(dear\s+[^,，]+[,，])\s*(.+)$", opening, flags=_re.IGNORECASE)
    if m:
        opening_line = m.group(1).strip()
        remainder = m.group(2).strip()
        opening = opening_line
        if remainder:
            body = (remainder + ("\n\n" + body if body else "")).strip()

    # If closing accidentally contains the signature (e.g., "Thanks, Emily"),
    # split it so that signature is on its own line.
    if not signature:
        valedictions = ["thanks", "thank you", "best", "regards", "kind regards", "sincerely"]
        lc = closing.lower()
        for v in valedictions:
            if lc.startswith(v):
                # Try to capture a trailing name after comma
                mv = _re.match(rf"^\s*{_re.escape(v)}[,，]?\s+(.+)$", lc, flags=_re.IGNORECASE)
                if mv:
                    name_part = draft.get("signature") or mv.group(1).strip()
                    if name_part and 1 <= len(name_part) <= 64:
                        signature = name_part
                        # Normalize closing to have a comma
                        closing = v.title() + ","
                break

    lines: List[str] = []
    if channel == "email" and subject:
        lines.append(f"Subject: {subject}")
    if opening:
        lines.append(opening)
    if body:
        if opening:
            lines.append("")  # Blank line after greeting
        lines.append(body)
    # Closing + signature
    if closing or signature:
        if body:
            lines.append("")  # Blank line before closing
        if closing:
            lines.append(closing)
        if signature:
            # Single line break after closing, not an extra blank paragraph
            lines.append(signature)

    return "\n".join(lines)

def _compose_fallback_email(user: Dict[str, Any], target: Dict[str, Any], brief: Dict[str, Any]) -> str:
    """Fallback: generate a concise, human voice email when the model fails."""
    subject = f"Quick hello about {target.get('company') or target.get('title') or 'your work'}"
    tname_raw = (target.get('name') or target.get('full_name') or target.get('first_name') or '').strip()
    tfirst = tname_raw.split()[0] if tname_raw else ''
    opening = f"Dear {tfirst or '[Recipient Name]'},"
    overlaps = brief.get("overlaps") or []
    hook = (overlaps[0] if overlaps else f"Noticed your work at {target.get('company', 'your team')} — inspiring.")
    sender_name = user.get('name') or 'A fellow practitioner'
    sender_company = user.get('company')
    body = (
        f"I'm {sender_name}{(' at ' + sender_company) if sender_company else ''}. "
        f"{hook} I'd love to swap notes on {target.get('title') or 'recent projects'} and share a couple of takeaways."
    )
    closing = brief.get('call_to_action') or 'Open to a 10–15 min chat next week?'
    signature = sender_name
    return _compose_text_from_draft({
        'channel': 'email',
        'subject': subject,
        'opening': opening,
        'body': body,
        'closing': closing,
        'signature': signature,
    })


def _postprocess_and_personalize_content(content: Any, state: OutreachState) -> Dict[str, Any]:
    """Best-effort fixup: ensure non-empty messages and replace placeholders with actual facts.

    Handles both dict-based and Pydantic-based message objects.
    """
    if not isinstance(content, dict):
        # Try to convert Pydantic model to dict
        def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
            if isinstance(obj, dict):
                return obj
            try:
                return obj.model_dump()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                return obj.dict()  # type: ignore[attr-defined]
            except Exception:
                pass
            return None

        content = _to_dict(content) or {}

    def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            return obj
        try:
            return obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
        return None

    msgs: List[Dict[str, Any]] = []
    raw_msgs = content.get("messages") if isinstance(content, dict) else None
    if isinstance(raw_msgs, list):
        for m in raw_msgs:
            m_dict = _to_dict(m)
            if isinstance(m_dict, dict):
                msgs.append(m_dict)

    user = state.get("user_profile", {}) or {}
    target = state.get("target_record", {}) or {}
    intent = state.get("intent_to_connect", "")
    company = target.get("company") or target.get("company_name") or ""
    title = target.get("title") or target.get("job_title") or ""
    school = target.get("school") or target.get("education") or ""
    name_raw = (target.get("name") or target.get("full_name") or target.get("first_name") or "")
    name_str = str(name_raw).strip()
    first_name = name_str.split()[0] if name_str else ""
    # Use placeholder if no real name is available
    greet_name = first_name or "[Recipient Name]"
    sender_raw = (user.get("name") or "")
    sender_str = sender_raw.strip()
    sender = (sender_str.split()[0] if sender_str else "Friend")

    # Helpers to de-AI-ify phrasing
    def _format_replacement(rep: str, nm: str, sd: str) -> str:
        return rep.replace("{name}", nm or "there").replace("{sender}", sd or "")

    def _rephrase_banned_phrases(text: str, nm: str, sd: str) -> str:
        import re
        REPHRASE_MAP = {
            "i hope this message finds you well": "Hello {name}, I wanted to connect about…",
            "i hope you're doing well": "Good to connect with you",
            "i wanted to reach out": "I'm writing because…",
            "i am reaching out to connect": "I'm contacting you regarding…",
            "i came across your profile": "I saw your work and wanted to connect",
            "as an ai language model": "",
            "as a highly skilled professional": "",
            "to whom it may concern": "Dear {name}",
            "allow me to introduce myself": "I'm {sender}, and I work in…",
            "i am writing to you": "Just a quick note to…",
            "in today’s fast-paced world": "",
            "furthermore": "Also",
            "additionally": "Plus",
            "i appreciate your prompt reply": "Looking forward to your thoughts",
            "if you have any questions, please let me know": "Feel free to ask if you'd like more info",
            "thank you for your time and consideration": "Thanks for reviewing my note",
            "i look forward to hearing from you": "Hope to hear from you soon",
            "please do not hesitate to contact me": "You can reach me anytime",
            "best regards": "Thanks",
            "sincerely yours": "All the best",
            "with kind regards": "Regards",
            "i would like to inquire about": "Could you tell me more about",
            "please find attached": "I've attached",
            "kindly note": "Just a heads-up",
            "quick hello": "Hello",
            "swap notes": "compare ideas",
            "fellow practitioner": "colleague",
        }
        def preserve_case(matched: str, rep: str) -> str:
            if matched.isupper():
                return rep.upper()
            if matched[:1].isupper():
                return rep[:1].upper() + rep[1:]
            return rep
        out = text or ""
        for phrase, repl in REPHRASE_MAP.items():
            pattern = r"\b" + re.escape(phrase) + r"\b"
            regex = re.compile(pattern, flags=re.IGNORECASE)
            def _sub(m):
                prepared = _format_replacement(repl, nm, sd)
                return preserve_case(m.group(0), prepared)
            out = regex.sub(_sub, out)
        return out

    fixed_msgs: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            # Should not happen due to _to_dict above, but guard anyway
            m = _to_dict(m) or {}
        chan = (m.get("channel") or "").lower() or "email"
        subj = m.get("subject") or None
        opening = m.get("opening") or ""
        body = m.get("body") or ""
        closing = m.get("closing") or ""
        signature = m.get("signature") or sender

        # Replace common placeholders
        replacements = {
            "[Recipient's Name]": greet_name,
            "[Recipient Name]": greet_name,
            "[Target's Name]": greet_name,
            "[Your Name]": sender,
            "[Sender's Name]": sender,
        }
        for placeholder, real in replacements.items():
            opening = opening.replace(placeholder, real)
            body = body.replace(placeholder, real)
            closing = closing.replace(placeholder, real)
            signature = signature.replace(placeholder, real)

        # Normalize greeting: always "Dear {Name}," using real name or placeholder
        import re as _re
        orig_opening = opening.strip()
        if orig_opening:
            m = _re.match(r"^\s*(dear|hi|hello)\s+([^,，]+)[,，]\s*(.*)$", orig_opening, flags=_re.IGNORECASE)
            if m:
                remainder = m.group(3).strip()
                opening = f"Dear {greet_name},"
                if remainder:
                    body = (remainder + ("\n\n" + body if body else "")).strip()
            else:
                # Does not look like a greeting; move it into body and insert our greeting
                opening = f"Dear {greet_name},"
                body = (orig_opening + ("\n\n" + body if body else "")).strip()
        else:
            opening = f"Dear {greet_name},"

        # Subject line if missing (email only)
        if chan == "email" and (subj is None or not str(subj).strip()):
            focus = title or company or intent or "a quick hello"
            if company and title:
                focus = f"{title} at {company}"
            subj = f"Regarding {focus}"[:68]

        # Ensure subject includes an anchor when available
        if chan == "email" and subj:
            lowered = str(subj).lower()
            anchor = None
            if company and company.lower() not in lowered:
                anchor = company
            if (not anchor) and title and title.lower() not in lowered:
                anchor = title
            if anchor:
                subj = f"{subj} — {anchor}"[:68]

        # Light personalization injection if body is too generic
        hints = []
        if company:
            hints.append(f"your work at {company}")
        if title:
            hints.append(f"your role as {title}")
        if school:
            hints.append(f"our shared background at {school}")
        injected = ", and ".join(hints[:2])
        if injected and injected not in body:
            body = body + (" " if body and not body.endswith(" ") else "") + f"I’m especially interested in {injected}."

        # De-AI-ify wording across fields
        opening = _rephrase_banned_phrases(opening, first_name or greet_name, sender)
        body = _rephrase_banned_phrases(body, first_name or greet_name, sender)
        closing = _rephrase_banned_phrases(closing, first_name or greet_name, sender)
        if subj is not None:
            subj = _rephrase_banned_phrases(str(subj), first_name or greet_name, sender)
        signature = _rephrase_banned_phrases(signature, first_name or greet_name, sender)

        # Minimal personalization count: append one more fact sentence if needed
        personal_hits = 0
        lower_body = body.lower()
        if company and company.lower() in lower_body:
            personal_hits += 1
        if title and title.lower() in lower_body:
            personal_hits += 1
        if school and school.lower() in lower_body:
            personal_hits += 1
        if personal_hits < 3:
            add_bits = []
            if company and company.lower() not in lower_body:
                add_bits.append(company)
            if title and title.lower() not in lower_body:
                add_bits.append(title)
            if school and school.lower() not in lower_body:
                add_bits.append(school)
            if add_bits:
                body = body + (" " if body and not body.endswith(" ") else "") + f"I’m particularly interested in {', '.join(add_bits[:3])}."

        fixed_msgs.append({
            "channel": chan,
            "subject": subj,
            "opening": opening,
            "body": body,
            "closing": closing or ("Would you be open to a brief call next week?" if intent else ""),
            "signature": signature or sender,
        })

    if not fixed_msgs:
        # No usable messages produced
        content = {}
    else:
        content = {"rationale": content.get("rationale", "personalized fixup applied"), "messages": fixed_msgs}
    return content

def _save_messages_to_csv(
    *,
    request_id: int,
    ru_id: int,
    user_id: str,
    messages: Dict[int, str],
    intent_to_connect: str,
    output_path: str = OUTPUT_CSV_PATH,
) -> None:
    """Mirror the notebook's CSV writer to keep the same output format.
    Preserve the same columns and add a version tag for provenance.
    """

    rows = []
    for target_id, message in messages.items():  # Iterate targets and messages
        rows.append(
            {
                "request_id": request_id,
                "ru_id": ru_id,
                "user_id": user_id,
                "target_user_id": target_id,
                "intent_to_connect": intent_to_connect,
                "message": message,
                "version": VERSION_TAG,
            }
        )

    file_exists = Path(output_path).exists()  # Detect whether header is needed
    with open(output_path, mode="a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "request_id",
                "ru_id",
                "user_id",
                "target_user_id",
                "intent_to_connect",
                "message",
                "version",
            ],
        )
        if not file_exists:  # Write header for a new file
            writer.writeheader()
        writer.writerows(rows)

def _ensure_llm() -> BaseChatModel:
    """Create an LLM according to env settings.

    Supported providers via env vars:
      - LLM_PROVIDER: "groq" (default) or "openai" or "azure".
    """
    load_dotenv(find_dotenv())  # Load .env
    provider = (os.environ.get("LLM_PROVIDER") or os.environ.get("USE_MODEL") or "groq").strip().lower()
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.4"))  # Default temperature 0.4

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed. `pip install langchain-openai`. ")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider in {"azure", "azure-openai", "openai-azure"}:
        if AzureChatOpenAI is None:
            raise RuntimeError("AzureChatOpenAI not available. `pip install langchain-openai`. ")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError("Please set AZURE_OPENAI_DEPLOYMENT_NAME in .env")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            openai_api_version=api_version,
            temperature=temperature,
        )

    if provider == "groq":  # Groq (very fast inference)
        if ChatGroq is None:
            raise RuntimeError("langchain-groq not installed. `pip install langchain-groq`. ")
        model = os.environ.get("GROQ_MODEL_VERSION", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=temperature)

    raise RuntimeError("Unsupported LLM_PROVIDER. Set to 'groq' or 'openai'.")

def _status_from_messages(messages: Dict[int, str]) -> str:
    empty = sum(1 for m in messages.values() if not m)  # Count empty messages
    return "SUCCESS" if empty < len(messages) else "FAIL"  # Success if at least one non-empty

def run_sample_preview(request_id: int = 123, intent_override: Optional[str] = None) -> Dict[str, Any]:
    """Replicate the notebook preview using the LangGraph workflow.

    Returns a result dict with per-message timings under `Per_Message_Times`.
    """
    ru_id, user_id, targets, profiles = _prepare_sample_context()  # Prepare sample context

    chosen_intent = (intent_override or os.environ.get("INTENT_TO_CONNECT") or INTENT_TO_CONNECT)
    llm = _ensure_llm()  # Create LLM from env
    graph = build_outreach_graph(llm)  # Build LangGraph

    messages: Dict[int, str] = {}  # Generated messages
    per_times: Dict[int, float] = {}  # Per-message time cost

    overall_start = time.time()  # Start overall timer

    with tqdm(total=len(targets), desc="Targets", unit="tg") as pbar:  # Progress bar
        for target_id in targets:  # Iterate through targets
            target_profile = profiles.get(target_id, {})  # Target profile
            user_profile = profiles.get(ru_id, {})  # Sender profile

            def _cb(step: str) -> None:  # Progress callback
                pbar.set_postfix_str(f"T{target_id}:{step}")
                tqdm.write(f"[Target {target_id}] Step -> {step}")

            t0 = time.time()  # Start timer per target
            state = run_outreach_workflow(  # Run full workflow
                graph,
                user_profile=user_profile,
                target_record=target_profile,
                recommended_channels=["email"],  # Use email in sample
                progress_callback=_cb,
                intent_to_connect=chosen_intent,
            )
            dt = time.time() - t0  # Elapsed time per target
            per_times[target_id] = dt

            # Prefer draft_bundle to avoid cases where final_messages is missing/unconverted
            bundle = state.get("draft_bundle", {}) or {}
            drafts = []
            if isinstance(bundle, dict):
                maybe = bundle.get("messages")
                if isinstance(maybe, list):
                    drafts = maybe
            if not drafts:
                drafts = state.get("final_messages", []) or []  # Fallback to final_messages
            message_text = ""
            chosen = None
            if isinstance(drafts, list) and drafts:
                # Prefer the email draft; otherwise use the first
                for d in drafts:
                    if isinstance(d, dict) and (d.get("channel") or "").lower() == "email":
                        chosen = d
                        break
                if chosen is None:
                    chosen = drafts[0]
                if isinstance(chosen, dict):
                    message_text = _compose_text_from_draft(chosen)
            # Fallback only when there are no drafts
            if (not drafts) or (chosen is None):
                message_text = _compose_fallback_email(user_profile, target_profile, state.get("persona_brief", {}))
            messages[target_id] = message_text
            pbar.update(1)

    overall_time = time.time() - overall_start  # Total elapsed time

    _save_messages_to_csv(
        request_id=request_id,
        ru_id=ru_id,
        user_id=user_id,
        messages=messages,
        intent_to_connect=chosen_intent,
    )

    result = {
        "Request_ID": request_id,
        "Messages": messages,
        "Status": _status_from_messages(messages),
        "Query_Time": 0.0,
        "Message_Generation_Time": overall_time,
        "Per_Message_Times": per_times,
        "Empty_Message_Count": sum(1 for m in messages.values() if not m),
    }
    return result

def cli_main() -> None:
    """Simple CLI: ask user for an outreach intent, then run the sample flow.

    This matches a terminal-style input box experience: we print guidance,
    accept one free-text line for intent, and fall back to .env default if empty.
    """
    # Show guidance and examples (force flush in case stdout is buffered)
    print("=========================================")
    print("Input: Outreach Intent")
    print("- Type your purpose in one sentence (free text)")
    print("- Examples:")
    print("  • I saw your profile and would appreciate a referral for the Data Scientist position.")
    print("  • I'm seeking brief career advice about transitioning into MLE roles.")
    print("  • I'm exploring project collaboration on LLM evaluation tooling.")
    print("Press Enter to use default from .env (INTENT_TO_CONNECT).")
    print("=========================================\n", flush=True)

    try:
        # If stdin is not a TTY (e.g., run in a non-interactive console), skip prompt
        import sys as _sys
        if hasattr(_sys.stdin, "isatty") and not _sys.stdin.isatty():
            intent = ""
        else:
            intent = input("Intent: ").strip()
    except EOFError:
        intent = ""

    intent = intent or os.environ.get("INTENT_TO_CONNECT") or INTENT_TO_CONNECT

    try:
        out = run_sample_preview(intent_override=intent)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"Messages saved to {OUTPUT_CSV_PATH}.")

        # Ask to send the first generated email via Gmail OAuth
        try:
            messages: Dict[int, str] = out.get("Messages", {}) if isinstance(out, dict) else {}
        except Exception:
            messages = {}
        if messages:
            # Normalize keys to ints and pick the first
            try:
                first_tid = sorted(int(k) for k in messages.keys())[0]
            except Exception:
                first_tid = list(messages.keys())[0]
            msg_text = messages[first_tid]
            subject, body = _split_subject_body(msg_text)
            print("\n--- Outreach Message Preview (to be sent) ---")
            print(f"Subject: {subject}")
            print(body)
            print("-------------------------------------------\n")
            # Ask recipient only in interactive terminals; otherwise skip send
            import sys as _sys
            if hasattr(_sys.stdin, "isatty") and _sys.stdin.isatty():
                default_rcpt = os.environ.get("SEND_TO", "qianyu1010@qq.com")
                try:
                    rcpt = input(f"Recipient email [default: {default_rcpt}]: ").strip()
                except EOFError:
                    rcpt = ""
                rcpt = rcpt or default_rcpt

                ans = input(f"Send this email to {rcpt} now? [y/N]: ").strip().lower()
                if ans in ("y", "yes"):
                    try:
                        _send_email_via_gmail_oauth(
                            to_email=rcpt,
                            subject=subject or "Quick hello",
                            body=body or msg_text,
                        )
                        print("Sent successfully via Gmail OAuth.")
                    except Exception as send_exc:
                        print(f"Send failed: {send_exc}")
            else:
                print("Non-interactive environment detected; skipping send step.")
        else:
            print("No messages to send.")
    except Exception as exc:
        # Print full traceback to help diagnose environment/import issues
        import traceback as _tb  # local import to avoid top-level dependency
        print(f"Run failed: {exc}")
        _tb.print_exc()


def _split_subject_body(text: str) -> Tuple[str, str]:
    """Extract subject from first line starting with 'Subject:'; return (subject, body)."""
    lines = (text or "").splitlines()
    if lines and lines[0].lower().startswith("subject:"):
        subj = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
        return subj or "Quick hello", body
    return "Quick hello", (text or "").strip()


def _send_email_via_gmail_oauth(
    to_email: str,
    subject: str,
    body: str,
    *,
    from_email: Optional[str] = None,
    client_secrets_file: Optional[str] = None,
    token_file: Optional[str] = None,
) -> None:
    """Send an email using Gmail API with OAuth. Stores token locally after first consent."""
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    client_secrets_file = client_secrets_file or os.environ.get("GOOGLE_CLIENT_SECRETS", "google_client_secret.json")
    token_file = token_file or os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except Exception as dep_exc:
        raise RuntimeError(
            "Missing Google API packages. Install: pip install google-api-python-client google-auth google-auth-oauthlib"
        ) from dep_exc

    if not os.path.exists(client_secrets_file):
        raise FileNotFoundError(
            f"Google OAuth client secrets not found: {client_secrets_file}. "
            "Provide it via GOOGLE_CLIENT_SECRETS env/.env."
        )

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            # Choose OAuth method. Default to 'console' to avoid GUI/browser hangs.
            method = (os.environ.get("GMAIL_OAUTH_METHOD", "local").strip().lower() or "console")
            if method in ("local", "local_server"):
                print("Starting Google OAuth (local server). If browser does not open, set GMAIL_OAUTH_METHOD=console.")
                try:
                    creds = flow.run_local_server(port=0)
                except Exception:
                    print("Local-server OAuth failed; falling back to console flow.")
                    creds = flow.run_console()
            else:
                # Console flow: prints a URL and asks for the auth code in terminal
                print("Starting Google OAuth (console). A URL will be printed; paste the code back here.")
                creds = flow.run_console()
        with open(token_file, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)

    # Construct MIME and send (plain text only as requested)
    from email.message import EmailMessage
    from email.policy import SMTP as SMTPPolicy
    import base64

    # Use a high but valid RFC-compliant line length to avoid hard wraps
    policy = SMTPPolicy.clone(max_line_length=998)
    msg = EmailMessage(policy=policy)
    if from_email:
        msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body or "", subtype="plain", charset="utf-8", cte="8bit")

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()


if __name__ == "__main__":  
    cli_main()
