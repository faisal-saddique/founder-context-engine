"""
LLM client wrapper for content generation.
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from ...core.config import settings
from ...core.logging import logger


class LLMClient:
    """Wrapper for LLM operations."""
    
    def __init__(self):
        """Initialize LLM clients."""
        # Main generation model
        self.generation_model = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=settings.openai_api_key
        ) if settings.openai_api_key else None
        
        # Critique model
        self.critique_model = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=settings.openai_api_key
        ) if settings.openai_api_key else None
    
    async def generate_content(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate content using LLM.
        
        Args:
            system_prompt: System instructions
            user_prompt: User request
            context: Optional context (retrieved knowledge + rules)
            
        Returns:
            Generated content
        """
        if not self.generation_model:
            return "LLM not configured. Please set OPENAI_API_KEY."
        
        try:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if context:
                messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\n"})
            
            messages.append({"role": "user", "content": user_prompt})
            
            response = await self.generation_model.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    async def critique_content(
        self,
        content: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """
        Critique generated content.
        
        Args:
            content: Content to critique
            criteria: List of criteria to check
            
        Returns:
            Critique results
        """
        if not self.critique_model:
            return {"score": 0, "feedback": "Critique model not configured"}
        
        try:
            criteria_text = "\n".join(f"- {c}" for c in criteria)
            
            prompt = f"""Critique the following content based on these criteria:
{criteria_text}

CONTENT:
{content}

Provide a score (1-10) and specific feedback for each criterion."""
            
            response = await self.critique_model.ainvoke([
                {"role": "user", "content": prompt}
            ])
            
            return {
                "score": 8,  # Simplified - parse from response in production
                "feedback": response.content
            }
            
        except Exception as e:
            logger.error(f"Critique failed: {str(e)}")
            return {"score": 0, "feedback": f"Error: {str(e)}"}


# Global LLM client
llm_client = LLMClient()
