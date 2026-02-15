"""
LangGraph workflow for content generation.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from .state import GenerationState
from .nodes import (
    input_analysis_node,
    dual_retrieval_node,
    draft_generation_node,
    validation_node,
    critique_node
)
from ...core.logging import logger


def should_retry(state: GenerationState) -> Literal["draft_generation", "critique"]:
    """
    After validation, decide whether to retry or move on.
    Retries if validation failed and we haven't hit the cap.
    """
    if not state["validation_passed"] and state["retry_count"] < state["max_retries"]:
        return "draft_generation"
    return "critique"


class ContentGenerationWorkflow:
    """Manages the LangGraph workflow for content generation."""

    def __init__(self):
        """Initialize the workflow graph."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Flow:
        Input Analysis -> Dual Retrieval -> Draft Generation ->
        Validation --(pass or retries exhausted)--> Critique -> End
                   |-(fail, retries left)--------> Draft Generation (loop)
        """
        workflow = StateGraph(GenerationState)

        # Add nodes
        workflow.add_node("input_analysis", input_analysis_node)
        workflow.add_node("dual_retrieval", dual_retrieval_node)
        workflow.add_node("draft_generation", draft_generation_node)
        workflow.add_node("validation", validation_node)
        workflow.add_node("critique", critique_node)

        # Define edges
        workflow.set_entry_point("input_analysis")
        workflow.add_edge("input_analysis", "dual_retrieval")
        workflow.add_edge("dual_retrieval", "draft_generation")
        workflow.add_edge("draft_generation", "validation")

        # Conditional routing: retry on validation failure, proceed on pass
        workflow.add_conditional_edges("validation", should_retry)

        workflow.add_edge("critique", END)

        return workflow.compile()

    async def generate(
        self,
        platform: str,
        post_format: str = "general",
        tone: str = "professional",
        specific_resource_context: str = None,
        custom_instructions: str = None
    ) -> GenerationState:
        """
        Run the generation workflow.

        Args:
            platform: Target platform
            post_format: Type of post to generate
            tone: Tone of the content
            specific_resource_context: Optional specific resource to use
            custom_instructions: Optional custom instructions

        Returns:
            Final state with generated content
        """
        logger.info(f"Starting generation workflow for {platform}")

        # Initialize state
        initial_state = GenerationState(
            platform=platform,
            post_format=post_format,
            tone=tone,
            specific_resource_context=specific_resource_context,
            custom_instructions=custom_instructions,
            retrieved_knowledge=None,
            retrieved_rules=[],
            source_ids=[],
            draft_content=None,
            final_content=None,
            validation_passed=False,
            validation_issues=[],
            generation_id=None,
            rule_ids_applied=[],
            model_config={},
            critique_result=None,
            retry_count=0,
            max_retries=2,
            previous_issues=[],
            best_attempt=None,
        )

        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)

        logger.info(f"Workflow complete: {final_state['generation_id']}")

        return final_state


# Global workflow instance
content_workflow = ContentGenerationWorkflow()
