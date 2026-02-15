"""
Claim validator to detect and verify factual claims in generated content.
Prevents hallucinations by requiring source attribution.
"""
import re
from typing import List, Dict, Any, Tuple
from ...core.logging import logger
from ...models.schemas import ValidationIssue


class ClaimValidator:
    """
    Validates claims in generated content to prevent hallucinations.
    
    Claim Definition:
    - Numbers/percentages
    - Named entities (companies, people, products)
    - Superlatives (best, fastest, only, first, etc.)
    """
    
    # Pattern for detecting numbers and percentages
    NUMBER_PATTERN = r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b'
    
    # Pattern for superlatives
    SUPERLATIVE_PATTERN = r'\b(best|worst|fastest|slowest|largest|smallest|first|only|leading|top|premier|foremost)\b'
    
    # Pattern for named entities (simplified)
    COMPANY_INDICATORS = ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Technologies']
    
    def detect_claims(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect all claims in the content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of detected claims with metadata
        """
        claims = []
        
        # Detect numerical claims
        number_matches = re.finditer(self.NUMBER_PATTERN, content, re.IGNORECASE)
        for match in number_matches:
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            
            claims.append({
                "type": "numerical",
                "value": match.group(),
                "context": context.strip(),
                "position": match.start()
            })
        
        # Detect superlatives
        superlative_matches = re.finditer(self.SUPERLATIVE_PATTERN, content, re.IGNORECASE)
        for match in superlative_matches:
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end]
            
            claims.append({
                "type": "superlative",
                "value": match.group(),
                "context": context.strip(),
                "position": match.start()
            })
        
        # Detect company names (simple heuristic)
        for indicator in self.COMPANY_INDICATORS:
            pattern = rf'\b[A-Z][a-zA-Z]+\s+{indicator}\b'
            matches = re.finditer(pattern, content)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                claims.append({
                    "type": "named_entity",
                    "value": match.group(),
                    "context": context.strip(),
                    "position": match.start()
                })
        
        logger.info(f"Detected {len(claims)} potential claims in content")
        return claims
    
    def validate_against_sources(
        self,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate claims against available sources.
        
        Args:
            claims: List of detected claims
            sources: List of source documents used in generation
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check if we have canonical brand facts
        has_canonical = any(
            source.get("type") == "canonical_brand_facts"
            for source in sources
        )
        
        for claim in claims:
            # Check if claim appears in any source
            claim_verified = False
            low_trust_source = False
            
            for source in sources:
                source_content = source.get("content", "")
                trust_score = source.get("trust_score", "Medium")
                
                # Simple substring check (in production, use better matching)
                if claim["value"] in source_content:
                    claim_verified = True
                    if trust_score == "Low":
                        low_trust_source = True
                    break
            
            # Generate issues for unverified claims
            if not claim_verified:
                if claim["type"] == "numerical":
                    issues.append(ValidationIssue(
                        claim=claim["context"],
                        issue_type="unverified_number",
                        severity="high",
                        suggestion="Remove specific number or link to source"
                    ))
                elif claim["type"] == "superlative":
                    issues.append(ValidationIssue(
                        claim=claim["context"],
                        issue_type="unverified_superlative",
                        severity="high",
                        suggestion="Replace with verified statement or remove superlative"
                    ))
                elif claim["type"] == "named_entity":
                    issues.append(ValidationIssue(
                        claim=claim["context"],
                        issue_type="unverified_entity",
                        severity="medium",
                        suggestion="Verify entity name against sources"
                    ))
            
            # Warn about low-trust sources
            elif low_trust_source:
                issues.append(ValidationIssue(
                    claim=claim["context"],
                    issue_type="low_trust_source",
                    severity="medium",
                    suggestion="Consider rewording to be more general"
                ))
        
        is_valid = len([i for i in issues if i.severity == "high"]) == 0
        
        logger.info(f"Validation complete: {'PASS' if is_valid else 'FAIL'} ({len(issues)} issues)")
        return is_valid, issues
    
    def validate_content(
        self,
        content: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Full validation pipeline: detect claims and validate against sources.
        
        Args:
            content: Generated content to validate
            sources: Source documents used in generation
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        # Detect all claims
        claims = self.detect_claims(content)
        
        # Validate against sources
        is_valid, issues = self.validate_against_sources(claims, sources)
        
        return is_valid, issues


# Global validator instance
claim_validator = ClaimValidator()
