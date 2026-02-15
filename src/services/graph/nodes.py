"""
Individual nodes for the LangGraph workflow.
"""
from typing import Dict, Any, List
import uuid

from .state import GenerationState
from ..retrieval.lightrag_client import lightrag_client
from ..retrieval.rule_retriever import rule_retriever
from ..llm.client import llm_client
from ..validation.claim_validator import claim_validator
from ...core.logging import logger


def _format_correction_prompt(issues: List[Dict[str, Any]]) -> str:
    """
    Build a correction block that tells the LLM exactly what to fix.
    Each issue includes the claim text, what was wrong, and a suggested fix.
    """
    type_labels = {
        "unverified_number": "Unverified number",
        "unverified_superlative": "Unverified superlative",
        "unverified_entity": "Unverified company/entity name",
        "low_trust_source": "Claim from low-trust source",
    }

    lines = [
        "\n\nIMPORTANT - YOUR PREVIOUS DRAFT HAD VALIDATION ISSUES. FIX ALL OF THESE:",
    ]
    for i, issue in enumerate(issues, 1):
        claim = issue.get("claim", "unknown claim")
        issue_type = issue.get("issue_type", "unknown")
        suggestion = issue.get("suggestion", "Remove or rephrase")
        severity = issue.get("severity", "unknown")
        label = type_labels.get(issue_type, issue_type)

        lines.append(f"{i}. [{severity.upper()}] {label}: \"{claim}\"")
        lines.append(f"   Fix: {suggestion}")

    lines.append("\nDo NOT include any claims that cannot be verified from the provided context.")
    lines.append("Only use facts, numbers, and names that appear in the CONTEXT above.")

    return "\n".join(lines)


async def input_analysis_node(state: GenerationState) -> GenerationState:
    """
    Node 1: Analyze input and prepare for retrieval.
    """
    logger.info("Node 1: Input Analysis")

    # Generate a unique ID for this generation
    state["generation_id"] = str(uuid.uuid4())
    state["source_ids"] = []
    state["rule_ids_applied"] = []
    state["model_config"] = {
        "model": "gpt-4",
        "temperature": 0.7
    }

    return state


async def dual_retrieval_node(state: GenerationState) -> GenerationState:
    """
    Node 2: Retrieve knowledge from LightRAG and rules from PostgreSQL.
    """
    logger.info("Node 2: Dual Retrieval")

    # Construct query from platform and format
    query = f"Generate a {state['post_format']} post for {state['platform']}"
    if state.get("specific_resource_context"):
        query += f" based on: {state['specific_resource_context']}"

    # Retrieve knowledge from LightRAG
    try:
        knowledge = await lightrag_client.query_knowledge(
            query=query,
            mode="hybrid",
            top_k=8,
            chunk_top_k=20
        )
        state["retrieved_knowledge"] = knowledge
    except Exception as e:
        logger.warning(f"Knowledge retrieval failed: {str(e)}")
        state["retrieved_knowledge"] = "No knowledge retrieved"

    # Retrieve rules from PostgreSQL
    try:
        rules = await rule_retriever.get_active_rules(
            platform=state["platform"],
            include_global=True
        )
        state["retrieved_rules"] = rules
        state["rule_ids_applied"] = [r["id"] for r in rules]
    except Exception as e:
        logger.warning(f"Rule retrieval failed: {str(e)}")
        state["retrieved_rules"] = []

    return state


async def draft_generation_node(state: GenerationState) -> GenerationState:
    """
    Node 3: Generate draft content using LLM.
    On retries, includes specific feedback about previous validation failures.
    """
    is_retry = state.get("retry_count", 0) > 0 and len(state.get("previous_issues", [])) > 0

    if is_retry:
        logger.info(f"Node 3: Draft Generation (retry {state['retry_count']})")
    else:
        logger.info("Node 3: Draft Generation")

    # Build system prompt with platform instructions
    system_prompt = f"""You are a professional content generator for {state['platform']}.
Generate high-quality, brand-safe content following these rules:

RULES:
"""

    # Add retrieved rules
    for rule in state["retrieved_rules"]:
        system_prompt += f"- {rule['constraint_text']}\n"

    # Add custom instructions
    if state.get("custom_instructions"):
        system_prompt += f"\nADDITIONAL INSTRUCTIONS:\n{state['custom_instructions']}\n"

    # Build user prompt
    user_prompt = f"""Create a {state['post_format']} post for {state['platform']} with a {state['tone']} tone.

CONTEXT (USE THIS INFORMATION):
{state['retrieved_knowledge']}

Generate engaging content that follows all rules and uses the provided context."""

    # On retry, append correction instructions so the LLM knows what went wrong
    if is_retry:
        user_prompt += _format_correction_prompt(state["previous_issues"])

    # Generate content
    draft = await llm_client.generate_content(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    state["draft_content"] = draft

    return state


async def validation_node(state: GenerationState) -> GenerationState:
    """
    Node 4: Validate claims in generated content.
    Tracks the best attempt across retries and prepares state for retry or finalization.
    """
    attempt = state.get("retry_count", 0) + 1
    logger.info(f"Node 4: Claim Validation (attempt {attempt})")

    # Prepare sources for validation
    sources = [{
        "content": state["retrieved_knowledge"],
        "trust_score": "High",
        "type": "retrieved_knowledge"
    }]

    # Validate content
    is_valid, issues = claim_validator.validate_content(
        content=state["draft_content"],
        sources=sources
    )

    state["validation_passed"] = is_valid
    state["validation_issues"] = [issue.model_dump() for issue in issues]

    current_issue_count = len(issues)

    # Track the best attempt we've seen (fewest issues)
    if state.get("best_attempt") is None or current_issue_count < state["best_attempt"]["issue_count"]:
        state["best_attempt"] = {
            "content": state["draft_content"],
            "issue_count": current_issue_count,
        }

    if is_valid:
        # Clean pass — use this draft
        state["final_content"] = state["draft_content"]
    elif state.get("retry_count", 0) < state.get("max_retries", 2):
        # Will retry — stash issues for the next generation pass
        state["previous_issues"] = state["validation_issues"]
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(
            f"Validation found {current_issue_count} issues, "
            f"retrying ({state['retry_count']}/{state.get('max_retries', 2)})"
        )
    else:
        # Retries exhausted — use whichever attempt had the fewest issues
        state["final_content"] = state["best_attempt"]["content"]
        logger.warning(
            f"Retries exhausted. Using best attempt "
            f"({state['best_attempt']['issue_count']} issues vs "
            f"{current_issue_count} on final try)"
        )

    return state


async def critique_node(state: GenerationState) -> GenerationState:
    """
    Node 5: Critique and log the generation.
    """
    logger.info("Node 5: Critique & Logging")

    # Safety net: ensure final_content is always set before critiquing
    if not state.get("final_content"):
        best = state.get("best_attempt")
        if best and best.get("content"):
            state["final_content"] = best["content"]
        elif state.get("draft_content"):
            state["final_content"] = state["draft_content"]
        else:
            state["final_content"] = "Content generation failed. Please try again."
        logger.warning("final_content was empty, used fallback")

    # Critique the content
    criteria = [
        "Follows platform guidelines",
        "Maintains professional tone",
        "Uses provided context accurately",
        "No unsupported claims",
        "Engaging and well-structured"
    ]

    critique = await llm_client.critique_content(
        content=state["final_content"],
        criteria=criteria
    )

    state["critique_result"] = critique

    logger.info(f"Generation complete: {state['generation_id']}")

    return state
