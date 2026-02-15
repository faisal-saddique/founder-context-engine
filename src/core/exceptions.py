"""
Custom exceptions for the application.
Provides specific error types for better error handling.
"""


class FounderContextEngineError(Exception):
    """Base exception for all application errors."""
    pass


class ParsingError(FounderContextEngineError):
    """Raised when document/URL parsing fails."""
    pass


class IngestionError(FounderContextEngineError):
    """Raised when data ingestion to LightRAG fails."""
    pass


class RetrievalError(FounderContextEngineError):
    """Raised when retrieval from LightRAG or Postgres fails."""
    pass


class ValidationError(FounderContextEngineError):
    """Raised when claim validation fails."""
    pass


class GenerationError(FounderContextEngineError):
    """Raised when content generation fails."""
    pass


class DatabaseError(FounderContextEngineError):
    """Raised when database operations fail."""
    pass


class ExternalServiceError(FounderContextEngineError):
    """Raised when external API calls fail."""
    pass
