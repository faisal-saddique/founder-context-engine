"""
Content classifier (the "Librarian") that maps raw text into one of 5 master schemas.
Uses gpt-4o-mini with structured output for guaranteed schema conformance.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ...core.config import settings
from ...core.logging import logger
from ...models.schemas import ContentSchema

CLASSIFICATION_PROMPT = """You are a content librarian. Your job is to classify incoming text into exactly one of 5 knowledge schemas.

## The 5 Schemas

1. **case_study** — Apps, client projects, portfolio items, internal product performance summaries.
   Signals: product names, metrics (ARR, MRR, downloads, growth), problem/solution structure, results.

2. **profile** — ICPs (Ideal Customer Profiles), user personas, competitor profiles, client archetypes.
   Signals: persona names, pain points, goals/aspirations, buying triggers, objections, "why we're a fit".

3. **guide** — Playbooks, SOPs, platform rules, proposal style guidelines, internal best practices.
   Signals: step-by-step instructions, do/don't lists, principles, guardrails, formatting rules.

4. **market_intel** — Discord news, YouTube summaries, industry updates, trends, platform changes.
   Signals: dates, source references, "what changed", strategic implications, recency language.

5. **general** — Ambiguous or mixed content that doesn't clearly fit the above categories.
   Use this only when none of the other 4 schemas fit with reasonable confidence.

## Rules
- Pick the SINGLE best-fit schema.
- Assign a confidence score from 0 to 100.
- If the content mixes multiple types, pick the dominant one."""


class ClassificationResult(BaseModel):
    """Structured output schema for the classifier LLM."""
    schema_type: Literal["case_study", "profile", "guide", "market_intel", "general"] = Field(
        description="The best-fit content schema"
    )
    confidence: float = Field(
        description="Confidence score from 0 to 100",
        ge=0,
        le=100,
    )


class ContentClassifier:
    """Classifies content into one of the 5 master schemas using an LLM with structured output."""

    def __init__(self):
        if settings.openai_api_key:
            base_model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=settings.openai_api_key,
            )
            self.chain = (
                ChatPromptTemplate.from_messages([
                    ("system", CLASSIFICATION_PROMPT),
                    ("user", "{content}"),
                ])
                | base_model.with_structured_output(ClassificationResult)
            )
        else:
            self.chain = None

    async def classify(self, markdown_text: str) -> tuple[ContentSchema, float]:
        """
        Classify content and return (schema, confidence).
        Falls back to GENERAL if confidence < 60 or on any error.
        """
        if not self.chain:
            logger.warning("No OpenAI key configured, defaulting to general schema")
            return ContentSchema.GENERAL, 0.0

        try:
            # trim to ~4k chars to keep token usage low
            snippet = markdown_text[:4000]

            result: ClassificationResult = await self.chain.ainvoke({"content": snippet})

            schema = ContentSchema(result.schema_type)
            confidence = result.confidence

            # apply the 60% threshold
            if confidence < 60:
                logger.info(f"Low confidence ({confidence}%), falling back to general")
                return ContentSchema.GENERAL, confidence

            logger.info(f"Classified as {schema.value} with {confidence}% confidence")
            return schema, confidence

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ContentSchema.GENERAL, 0.0


# global instance
content_classifier = ContentClassifier()
