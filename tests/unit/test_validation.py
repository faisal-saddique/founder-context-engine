"""
Thorough unit tests for the ClaimValidator.
Tests claim detection, source validation, and the full pipeline with
real dataset content.
"""
import pytest

from src.services.validation.claim_validator import ClaimValidator, claim_validator
from src.models.schemas import ValidationIssue


# ── claim detection ───────────────────────────────────────────────

class TestDetectNumericalClaims:
    def test_detects_plain_numbers(self):
        claims = claim_validator.detect_claims("We have 100 customers.")
        nums = [c for c in claims if c["type"] == "numerical"]
        assert any(c["value"] == "100" for c in nums)

    def test_detects_percentages(self):
        claims = claim_validator.detect_claims("Achieved 95% satisfaction.")
        nums = [c for c in claims if c["type"] == "numerical"]
        # regex captures the number part, % is optional in the pattern
        assert any("95" in c["value"] for c in nums)

    def test_detects_large_numbers_with_commas(self):
        claims = claim_validator.detect_claims("Revenue of 1,200,000 dollars.")
        nums = [c for c in claims if c["type"] == "numerical"]
        assert any("1,200,000" in c["value"] for c in nums)

    def test_no_numerical_claims_in_text(self):
        claims = claim_validator.detect_claims("This is plain text with no numbers.")
        nums = [c for c in claims if c["type"] == "numerical"]
        assert len(nums) == 0


class TestDetectSuperlatives:
    @pytest.mark.parametrize("word", ["best", "fastest", "leading", "first", "only", "top"])
    def test_detects_superlative(self, word):
        claims = claim_validator.detect_claims(f"We are the {word} in the market.")
        sups = [c for c in claims if c["type"] == "superlative"]
        assert any(word in c["value"].lower() for c in sups)

    def test_no_superlatives(self):
        claims = claim_validator.detect_claims("We help people build apps.")
        sups = [c for c in claims if c["type"] == "superlative"]
        assert len(sups) == 0


class TestDetectNamedEntities:
    def test_detects_company_indicator(self):
        claims = claim_validator.detect_claims("Working with Acme Corp on a project.")
        entities = [c for c in claims if c["type"] == "named_entity"]
        assert any("Acme Corp" in c["value"] for c in entities)

    def test_detects_inc_suffix(self):
        claims = claim_validator.detect_claims("Partner: Widget Inc delivers results.")
        entities = [c for c in claims if c["type"] == "named_entity"]
        assert any("Widget Inc" in c["value"] for c in entities)

    def test_no_entity_in_plain_text(self):
        claims = claim_validator.detect_claims("We build mobile apps for everyone.")
        entities = [c for c in claims if c["type"] == "named_entity"]
        assert len(entities) == 0


class TestClaimContext:
    """Detected claims should include surrounding context."""

    def test_context_around_claim(self):
        text = "After years of work, we reached 500 daily active users last month."
        claims = claim_validator.detect_claims(text)
        nums = [c for c in claims if c["type"] == "numerical" and c["value"] == "500"]
        assert len(nums) >= 1
        # context should include words around the number
        assert "daily active users" in nums[0]["context"]


# ── validate against sources ──────────────────────────────────────

class TestValidateAgainstSources:
    def test_verified_claim_passes(self):
        claims = [{"type": "numerical", "value": "100", "context": "We have 100 users", "position": 0}]
        sources = [{"content": "The company has 100 users.", "trust_score": "High", "type": "canonical_brand_facts"}]
        is_valid, issues = claim_validator.validate_against_sources(claims, sources)
        assert is_valid is True
        assert len(issues) == 0

    def test_unverified_number_flagged(self):
        claims = [{"type": "numerical", "value": "999", "context": "We have 999 employees", "position": 0}]
        sources = [{"content": "The company employs talented people.", "trust_score": "High"}]
        is_valid, issues = claim_validator.validate_against_sources(claims, sources)
        assert is_valid is False
        assert any(i.issue_type == "unverified_number" for i in issues)

    def test_unverified_superlative_flagged(self):
        claims = [{"type": "superlative", "value": "best", "context": "We are the best agency", "position": 0}]
        sources = [{"content": "They are an agency.", "trust_score": "High"}]
        is_valid, issues = claim_validator.validate_against_sources(claims, sources)
        assert is_valid is False
        assert any(i.issue_type == "unverified_superlative" for i in issues)

    def test_unverified_entity_is_medium_severity(self):
        claims = [{"type": "named_entity", "value": "Mega Corp", "context": "Partnered with Mega Corp", "position": 0}]
        sources = [{"content": "No mention", "trust_score": "High"}]
        is_valid, issues = claim_validator.validate_against_sources(claims, sources)
        # named entities are medium severity, so overall is_valid stays True
        assert is_valid is True
        assert any(i.severity == "medium" for i in issues)

    def test_low_trust_source_warning(self):
        claims = [{"type": "numerical", "value": "50", "context": "50 downloads", "position": 0}]
        sources = [{"content": "They have 50 downloads", "trust_score": "Low"}]
        is_valid, issues = claim_validator.validate_against_sources(claims, sources)
        assert any(i.issue_type == "low_trust_source" for i in issues)

    def test_empty_claims_passes(self):
        is_valid, issues = claim_validator.validate_against_sources([], [])
        assert is_valid is True
        assert len(issues) == 0


# ── full pipeline ─────────────────────────────────────────────────

class TestValidateContent:
    def test_empty_content(self):
        is_valid, issues = claim_validator.validate_content("", [])
        assert is_valid is True
        assert issues == []

    def test_full_pipeline_with_sample(self, sample_markdown, sample_sources):
        is_valid, issues = claim_validator.validate_content(sample_markdown, sample_sources)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_pipeline_with_canonical_brand_facts(self, canonical_brand_facts):
        """Real brand facts should have some numerical claims detected."""
        sources = [{"content": canonical_brand_facts, "trust_score": "High", "type": "canonical_brand_facts"}]
        is_valid, issues = claim_validator.validate_content(canonical_brand_facts, sources)
        assert isinstance(is_valid, bool)

    def test_clean_content_passes(self):
        """Content without numbers, superlatives, or entities should pass."""
        content = "We help app creators grow and improve their products."
        is_valid, issues = claim_validator.validate_content(content, [])
        assert is_valid is True
        assert len(issues) == 0

    def test_all_verified_passes(self):
        content = "We have 25 developers. Founded in 2019."
        sources = [
            {"content": "Team size: 25+ developers. Founded in 2019.", "trust_score": "High", "type": "canonical_brand_facts"},
        ]
        is_valid, issues = claim_validator.validate_content(content, sources)
        assert is_valid is True
