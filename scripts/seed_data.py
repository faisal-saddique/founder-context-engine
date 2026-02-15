"""
Seed script to populate databases with test data.
"""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from prisma import Prisma


async def seed_data():
    """Populate database with test data."""
    try:
        print("Connecting to PostgreSQL...")
        db = Prisma()
        await db.connect()
        
        # Seed platform configs
        print("\nSeeding platform configs...")
        platforms = [
            {"platformName": "linkedin", "customInstructions": "Keep posts under 1300 characters. Use professional tone. Include relevant hashtags."},
            {"platformName": "twitter", "customInstructions": "Keep under 280 characters. Be punchy and engaging. Use 2-3 hashtags max."},
            {"platformName": "upwork", "customInstructions": "Focus on client value. Be specific about deliverables. Professional but approachable."}
        ]
        
        for p in platforms:
            existing = await db.platformconfig.find_unique(where={"platformName": p["platformName"]})
            if not existing:
                await db.platformconfig.create(data=p)
                print(f"  ✓ Created: {p['platformName']}")
            else:
                print(f"  - Exists: {p['platformName']}")
        
        # Seed agent rules
        print("\nSeeding agent rules...")
        rules = [
            {"platform": "global", "scope": "GLOBAL", "constraintText": "Never fabricate metrics or statistics", "ruleType": "negative", "priority": 100, "status": "ACTIVE"},
            {"platform": "global", "scope": "GLOBAL", "constraintText": "Always maintain professional tone", "ruleType": "positive", "priority": 90, "status": "ACTIVE"},
            {"platform": "global", "scope": "GLOBAL", "constraintText": "Back all claims with sources", "ruleType": "positive", "priority": 95, "status": "ACTIVE"},
            {"platform": "linkedin", "scope": "PLATFORM", "constraintText": "Use thought leadership framing", "ruleType": "positive", "priority": 80, "status": "ACTIVE"},
            {"platform": "linkedin", "scope": "PLATFORM", "constraintText": "Include a call-to-action", "ruleType": "positive", "priority": 70, "status": "ACTIVE"},
            {"platform": "twitter", "scope": "PLATFORM", "constraintText": "Be concise and punchy", "ruleType": "positive", "priority": 80, "status": "ACTIVE"},
            {"platform": "twitter", "scope": "PLATFORM", "constraintText": "Use thread format for long content", "ruleType": "positive", "priority": 75, "status": "ACTIVE"},
            {"platform": "upwork", "scope": "PLATFORM", "constraintText": "Focus on client outcomes", "ruleType": "positive", "priority": 80, "status": "ACTIVE"},
            {"platform": "upwork", "scope": "PLATFORM", "constraintText": "Include relevant experience", "ruleType": "positive", "priority": 75, "status": "ACTIVE"},
        ]
        
        for r in rules:
            await db.agentrule.create(data=r)
            print(f"  ✓ Created: {r['constraintText'][:40]}...")
        
        # Seed sample generation
        print("\nSeeding sample generation...")
        from prisma import Json
        gen = await db.generation.create(
            data={
                "generationReason": "Test generation for LinkedIn",
                "promptVersion": "1.0",
                "finalOutput": "This is a sample generated post for testing purposes.",
                "sourceIdsUsed": Json(["test-source-1", "test-source-2"]),
                "ruleIdsApplied": Json([]),
                "modelConfig": Json({"model": "gpt-4", "temperature": 0.7})
            }
        )
        print(f"  ✓ Created generation: {gen.id}")
        
        # Seed feedback
        print("\nSeeding sample feedback...")
        feedback = await db.feedbacklog.create(
            data={
                "generationId": gen.id,
                "rating": 4,
                "category": "tone",
                "comment": "Good content but could be more engaging"
            }
        )
        print(f"  ✓ Created feedback: {feedback.id}")
        
        print("\n✓ Seed data complete!")
        
        # Show counts
        platform_count = await db.platformconfig.count()
        rule_count = await db.agentrule.count()
        gen_count = await db.generation.count()
        feedback_count = await db.feedbacklog.count()
        
        print(f"\nDatabase counts:")
        print(f"  - Platform configs: {platform_count}")
        print(f"  - Agent rules: {rule_count}")
        print(f"  - Generations: {gen_count}")
        print(f"  - Feedback logs: {feedback_count}")
        
        await db.disconnect()
        return True
        
    except Exception as e:
        print(f"\n✗ Seed failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = asyncio.run(seed_data())
    sys.exit(0 if success else 1)
