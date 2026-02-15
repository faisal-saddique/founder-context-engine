"""
Configuration management using Pydantic Settings.
Loads environment variables and provides type-safe config access.
"""
from urllib.parse import urlparse, unquote
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    tavily_api_key: str
    llama_cloud_api_key: str
    firecrawl_api_key: str
    
    # Database
    database_url: str
    direct_url: str
    
    # Neo4j
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # LangSmith
    langsmith_api_key: Optional[str] = None
    langchain_tracing_v2: bool = True
    langchain_project: str = "founder-context-engine"
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    @property
    def lightrag_pg_config(self) -> dict[str, str]:
        """Parse the direct Supabase URL into individual PG params for LightRAG."""
        parsed = urlparse(self.direct_url)
        return {
            "POSTGRES_HOST": parsed.hostname or "localhost",
            "POSTGRES_PORT": str(parsed.port or 5432),
            "POSTGRES_USER": unquote(parsed.username or "postgres"),
            "POSTGRES_PASSWORD": unquote(parsed.password or ""),
            "POSTGRES_DATABASE": parsed.path.lstrip("/") or "postgres",
        }


# Global settings instance
settings = Settings()
