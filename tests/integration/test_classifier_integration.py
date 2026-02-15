"""
Integration tests for the Librarian classifier with a real LLM call.
Requires OPENAI_API_KEY to be set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.models.schemas import ContentSchema
from src.services.ingestion.classifier import ContentClassifier

# sample snippets representing each schema type
SAMPLES = {
    ContentSchema.CASE_STUDY: """
# QR Code Scanner Pro - Case Study
**Product:** QR Code Scanner Pro (iOS)
**Stage:** Growth
**Vertical:** Utility Apps

## Problem
Users needed a fast, reliable QR code scanner with no ads. Existing solutions were bloated.

## Solution
Built a minimal, privacy-focused QR scanner with instant scan and history.

## Results
- 150K+ downloads in first 6 months
- 4.8 star rating on App Store
- MRR grew from $500 to $4,200
- 12% week-over-week subscriber growth
""",
    ContentSchema.PROFILE: """
# Ideal Customer Profile: Indie App Founder
**Segment:** Solo iOS/Android developers with 1-5 published apps

## Pain Points
- Struggling with ASO and discoverability
- No budget for paid acquisition
- Overwhelmed by marketing alongside development

## Goals
- Reach 10K organic downloads/month
- Build sustainable subscription revenue
- Free up time from marketing to focus on code

## Buying Triggers
- Just launched a new app and need visibility
- Seeing competitors outrank them in search

## Why We're a Fit
- We specialize in bootstrapped app growth
- Our playbooks are battle-tested on our own apps
""",
    ContentSchema.GUIDE: """
# ASO Playbook: App Store Optimization Best Practices

## Principles
1. Keywords first, branding second in the title
2. Update screenshots every major release
3. Localize for top 5 markets minimum

## Step-by-step
1. Research keywords using AppTweak or Sensor Tower
2. Place primary keyword in title (30 char limit)
3. Use subtitle for secondary keywords
4. Write description with natural keyword density

## Do
- A/B test screenshots quarterly
- Respond to every 1-star review within 24h

## Don't
- Keyword stuff the description
- Use competitor brand names in metadata
""",
    ContentSchema.MARKET_INTEL: """
# Apple Search Ads Changes - January 2026
**Source:** Apple Developer Blog
**Date:** January 15, 2026

## Key Takeaways
- Apple increased minimum bid for competitive categories by 40%
- New "Today" tab ad placements now available for all developers
- Privacy nutrition labels now affect ad delivery algorithms

## Strategic Implications
- Budget allocation needs to shift toward organic ASO
- "Today" tab placements could be valuable for launch campaigns

## What Changed
- Search Ads Advanced now requires iOS 17+ targeting
- Custom product pages can be linked to specific ad groups

## Why It Matters
- Cost per install likely to rise in competitive verticals
- Organic visibility becomes even more critical
""",
}


class TestClassifierIntegration:
    """Real LLM classification of sample content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("expected_schema,content", list(SAMPLES.items()))
    async def test_classifies_correctly(self, expected_schema, content):
        """Each sample should be classified into the correct schema with >= 60% confidence."""
        classifier = ContentClassifier()
        if classifier.chain is None:
            pytest.skip("OPENAI_API_KEY not set")

        schema, confidence = await classifier.classify(content)
        assert schema == expected_schema, f"Expected {expected_schema}, got {schema} ({confidence}%)"
        assert confidence >= 60, f"Confidence too low: {confidence}%"
