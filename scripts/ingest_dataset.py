"""
Ingest the full Dataset/ directory into LightRAG.
Reads markdown files directly, scrapes App Store links, extracts YouTube transcripts,
classifies each piece into a master schema, and commits to the knowledge base.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import logger
from src.models.schemas import ContentSchema
from src.models.knowledge import UnifiedKnowledge, KnowledgeMetadata
from src.services.ingestion.classifier import ContentClassifier
from src.services.retrieval.lightrag_client import LightRAGClient

DATASET_DIR = Path(__file__).parent.parent / "Dataset"

# files to skip
SKIP_FILES = {".DS_Store", "readme.md"}

# metadata mapping based on the dataset readme
# key = partial path match, value = metadata overrides
METADATA_HINTS = {
    "Canonical Brand Facts": {"trust_score": "High", "usage_permission": "public_safe", "content_type": "canonical_brand_facts"},
    "ICP": {"trust_score": "High", "usage_permission": "public_safe", "content_type": "icp"},
    "PlatformRules": {"trust_score": "High", "usage_permission": "public_safe", "content_type": "platform_rules"},
    "ProposalStyleGuidelines": {"trust_score": "High", "usage_permission": "internal_only", "content_type": "proposal_guidelines"},
    "playbooks": {"trust_score": "High", "usage_permission": "public_safe", "content_type": "playbook"},
    "industry_news": {"trust_score": "Medium", "usage_permission": "public_safe", "content_type": "market_intel"},
    "Apps/Cleaner": {"trust_score": "High", "usage_permission": "internal_only", "content_type": "app_case_study"},
    "Apps/Forms": {"trust_score": "High", "usage_permission": "internal_only", "content_type": "app_case_study"},
    "Apps/QR": {"trust_score": "High", "usage_permission": "internal_only", "content_type": "app_case_study"},
    "Apps/internal": {"trust_score": "High", "usage_permission": "internal_only", "content_type": "app_portfolio"},
    "Apps/App_Store": {"trust_score": "Medium", "usage_permission": "public_safe", "content_type": "app_store_listing"},
    "Youtube": {"trust_score": "Medium", "usage_permission": "public_safe", "content_type": "youtube_summary"},
}


def get_metadata_for_path(file_path: Path) -> dict:
    """Match file path against metadata hints."""
    rel = str(file_path.relative_to(DATASET_DIR))
    for pattern, meta in METADATA_HINTS.items():
        if pattern in rel:
            return meta
    return {"trust_score": "Medium", "usage_permission": "public_safe", "content_type": "general"}


async def ingest_markdown_file(
    file_path: Path,
    classifier: ContentClassifier,
    client: LightRAGClient,
    stats: dict,
):
    """Read a markdown file, classify it, and insert into LightRAG."""
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        print(f"  Skipping empty file: {file_path.name}")
        return

    schema, confidence = await classifier.classify(content)
    meta_hints = get_metadata_for_path(file_path)

    knowledge = UnifiedKnowledge(
        source_type="markdown",
        content_body=content,
        metadata=KnowledgeMetadata(
            author="Bilal",
            usage_permission=meta_hints["usage_permission"],
            trust_score=meta_hints["trust_score"],
            content_type=meta_hints["content_type"],
            content_schema=schema.value,
            file_name=file_path.name,
            tags=[],
        ),
    )

    lightrag_content = knowledge.to_lightrag_format()
    await client.insert_knowledge(content=lightrag_content, document_id=knowledge.id)

    stats["inserted"] += 1
    print(f"  [{schema.value:>12} {confidence:5.1f}%] {file_path.name}")


async def ingest_app_store_links(
    links_file: Path,
    classifier: ContentClassifier,
    client: LightRAGClient,
    stats: dict,
):
    """Scrape each App Store URL and ingest."""
    from src.services.ingestion.web_parser import WebParser
    parser = WebParser()

    urls = [line.strip() for line in links_file.read_text().splitlines() if line.strip()]
    for url in urls:
        try:
            markdown = await parser.parse(url)
            schema, confidence = await classifier.classify(markdown)

            knowledge = UnifiedKnowledge(
                source_type="app_store_link",
                content_body=markdown,
                metadata=KnowledgeMetadata(
                    author="Bilal",
                    usage_permission="public_safe",
                    trust_score="Medium",
                    content_type="app_store_listing",
                    content_schema=schema.value,
                    source_url=url,
                    tags=[],
                ),
            )
            lightrag_content = knowledge.to_lightrag_format()
            await client.insert_knowledge(content=lightrag_content, document_id=knowledge.id)
            stats["inserted"] += 1
            print(f"  [{schema.value:>12} {confidence:5.1f}%] {url}")
        except Exception as e:
            stats["failed"] += 1
            print(f"  [FAILED] {url}: {e}")


async def ingest_youtube_links(
    links_file: Path,
    classifier: ContentClassifier,
    client: LightRAGClient,
    stats: dict,
):
    """Extract transcripts from YouTube links and ingest."""
    from src.services.ingestion.video_parser import VideoParser
    parser = VideoParser()

    urls = [line.strip() for line in links_file.read_text().splitlines() if line.strip()]
    for url in urls:
        try:
            markdown = await parser.parse(url)
            schema, confidence = await classifier.classify(markdown)

            knowledge = UnifiedKnowledge(
                source_type="youtube_summary",
                content_body=markdown,
                metadata=KnowledgeMetadata(
                    author="Bilal",
                    usage_permission="public_safe",
                    trust_score="Medium",
                    content_type="youtube_summary",
                    content_schema=schema.value,
                    source_url=url,
                    tags=[],
                ),
            )
            lightrag_content = knowledge.to_lightrag_format()
            await client.insert_knowledge(content=lightrag_content, document_id=knowledge.id)
            stats["inserted"] += 1
            print(f"  [{schema.value:>12} {confidence:5.1f}%] {url}")
        except Exception as e:
            stats["failed"] += 1
            print(f"  [FAILED] {url}: {e}")


async def main():
    print("=" * 60)
    print("Dataset Ingestion")
    print("=" * 60)

    # init services
    classifier = ContentClassifier()
    client = LightRAGClient(working_dir="./rag_storage")
    await client.initialize()

    stats = {"inserted": 0, "failed": 0, "skipped": 0}

    # collect all markdown files
    md_files = sorted(DATASET_DIR.rglob("*.md"))
    print(f"\nFound {len(md_files)} markdown files")

    print("\n--- Markdown Files ---")
    for md_file in md_files:
        if md_file.name in SKIP_FILES:
            stats["skipped"] += 1
            continue
        try:
            await ingest_markdown_file(md_file, classifier, client, stats)
        except Exception as e:
            stats["failed"] += 1
            print(f"  [FAILED] {md_file.name}: {e}")

    # App Store links
    app_store_file = DATASET_DIR / "Apps" / "App_Store_Links.txt"
    if app_store_file.exists():
        print("\n--- App Store Links ---")
        await ingest_app_store_links(app_store_file, classifier, client, stats)

    # YouTube links
    youtube_file = DATASET_DIR / "Youtube" / "youtube_links.txt"
    if youtube_file.exists():
        print("\n--- YouTube Links ---")
        await ingest_youtube_links(youtube_file, classifier, client, stats)

    await client.finalize()

    print("\n" + "=" * 60)
    print(f"Done! Inserted: {stats['inserted']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
