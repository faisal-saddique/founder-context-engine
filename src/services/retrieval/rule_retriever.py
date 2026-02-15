"""
Rule retriever for fetching active rules from PostgreSQL.
"""
from typing import List, Dict, Any
from prisma import Prisma
from prisma.models import AgentRule

from ...core.logging import logger
from ...core.exceptions import RetrievalError


class RuleRetriever:
    """Retrieves rules from PostgreSQL based on platform and priority."""
    
    def __init__(self):
        """Initialize Prisma client."""
        self.db = Prisma(auto_register=True)
    
    async def connect(self):
        """Connect to database."""
        await self.db.connect()
        logger.info("Connected to PostgreSQL for rule retrieval")
    
    async def disconnect(self):
        """Disconnect from database."""
        await self.db.disconnect()
        logger.info("Disconnected from PostgreSQL")
    
    async def get_active_rules(
        self,
        platform: str,
        include_global: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve active rules for a specific platform.
        
        Args:
            platform: Target platform name
            include_global: Whether to include global rules
            
        Returns:
            List of rule dictionaries sorted by priority
        """
        try:
            logger.info(f"Retrieving rules for platform: {platform}")
            
            # Build filter conditions
            where_conditions = {
                "status": "ACTIVE"
            }
            
            # Query rules
            if include_global:
                rules = await self.db.agentrule.find_many(
                    where={
                        "status": "ACTIVE",
                        "OR": [
                            {"platform": platform},
                            {"scope": "GLOBAL"}
                        ]
                    },
                    order={"priority": "desc"}
                )
            else:
                rules = await self.db.agentrule.find_many(
                    where={
                        "status": "ACTIVE",
                        "platform": platform
                    },
                    order={"priority": "desc"}
                )
            
            # Convert to dictionaries
            rule_dicts = [
                {
                    "id": rule.id,
                    "platform": rule.platform,
                    "scope": rule.scope,
                    "constraint_text": rule.constraintText,
                    "rule_type": rule.ruleType,
                    "priority": rule.priority
                }
                for rule in rules
            ]
            
            logger.info(f"Retrieved {len(rule_dicts)} active rules for {platform}")
            return rule_dicts
            
        except Exception as e:
            logger.error(f"Failed to retrieve rules: {str(e)}")
            raise RetrievalError(f"Rule retrieval failed: {str(e)}")
    
    async def create_rule(
        self,
        platform: str,
        constraint_text: str,
        rule_type: str = None,
        priority: int = 10,
        scope: str = "PLATFORM",
        status: str = "PROPOSED"
    ) -> str:
        """
        Create a new rule (usually from feedback).
        
        Args:
            platform: Platform name
            constraint_text: The rule text
            rule_type: Type of rule (positive, negative, etc.)
            priority: Rule priority (higher = more important)
            scope: GLOBAL or PLATFORM
            status: PROPOSED, ACTIVE, or ARCHIVED
            
        Returns:
            Created rule ID
        """
        try:
            rule = await self.db.agentrule.create(
                data={
                    "platform": platform,
                    "scope": scope,
                    "constraintText": constraint_text,
                    "ruleType": rule_type,
                    "priority": priority,
                    "status": status
                }
            )
            
            logger.info(f"Created new rule: {rule.id}")
            return rule.id
            
        except Exception as e:
            logger.error(f"Failed to create rule: {str(e)}")
            raise RetrievalError(f"Rule creation failed: {str(e)}")


# Global rule retriever instance
rule_retriever = RuleRetriever()
