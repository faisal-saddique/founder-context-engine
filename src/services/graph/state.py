"""
State definitions for LangGraph workflow.
"""
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated


class GenerationState(TypedDict):
    """State for content generation workflow."""
    
    # Input
    platform: str
    post_format: str
    tone: str
    specific_resource_context: Optional[str]
    custom_instructions: Optional[str]
    
    # Retrieval
    retrieved_knowledge: Optional[str]
    retrieved_rules: List[Dict[str, Any]]
    source_ids: List[str]
    
    # Generation
    draft_content: Optional[str]
    final_content: Optional[str]
    
    # Validation
    validation_passed: bool
    validation_issues: List[Dict[str, Any]]

    # Retry tracking
    retry_count: int
    max_retries: int
    previous_issues: List[Dict[str, Any]]
    best_attempt: Optional[Dict[str, Any]]

    # Metadata
    generation_id: Optional[str]
    rule_ids_applied: List[str]
    model_config: Dict[str, Any]
    critique_result: Optional[Dict[str, Any]]
