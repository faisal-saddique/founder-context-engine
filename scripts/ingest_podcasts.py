"""
Ingest all Lenny's Podcast transcripts through the API endpoints.
Uses the proper parse_preview -> commit_ingest flow.
"""
import httpx
import asyncio
import os
import sys
import time
from pathlib import Path

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001")
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "Lenny's Podcast Transcripts Archive [public]"
# LightRAG has practical limits on document size, so we cap at 50k chars
MAX_CONTENT_LENGTH = 50_000
CONCURRENCY = 3
TIMEOUT = 180.0


async def ingest_one(client: httpx.AsyncClient, filepath: Path, semaphore: asyncio.Semaphore):
    """Run parse_preview then commit_ingest for a single transcript."""
    filename = filepath.name
    guest_name = filepath.stem

    content = filepath.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return {"file": filename, "status": "skipped", "reason": "empty file"}

    # Truncate very large transcripts to stay within practical limits
    if len(content) > MAX_CONTENT_LENGTH:
        content = content[:MAX_CONTENT_LENGTH]

    async with semaphore:
        # Step 1: parse_preview
        try:
            preview_resp = await client.post(
                f"{API_BASE}/api/v1/ingest/parse_preview",
                json={
                    "source_type": "markdown",
                    "content": content,
                    "metadata": {
                        "author": guest_name,
                        "type": "podcast_transcript",
                        "file_name": filename,
                        "tags": ["lenny_podcast", "transcript", guest_name.lower().replace(" ", "_")],
                    },
                },
                timeout=TIMEOUT,
            )
        except httpx.TimeoutException:
            return {"file": filename, "status": "error", "reason": "preview timeout"}
        except Exception as e:
            return {"file": filename, "status": "error", "reason": f"preview error: {e}"}

        if preview_resp.status_code != 200:
            return {
                "file": filename,
                "status": "error",
                "reason": f"preview {preview_resp.status_code}: {preview_resp.text[:200]}",
            }

        preview = preview_resp.json()
        schema = preview.get("content_schema", "general")
        confidence = preview.get("confidence_score", 0)

        # Step 2: commit_ingest
        try:
            commit_resp = await client.post(
                f"{API_BASE}/api/v1/ingest/commit_ingest",
                json={
                    "markdown_content": preview["markdown_content"],
                    "source_type": "markdown",
                    "content_schema": schema,
                    "metadata": {
                        "author": guest_name,
                        "type": "podcast_transcript",
                        "file_name": filename,
                        "source_url": "https://www.lennyspodcast.com/",
                        "tags": ["lenny_podcast", "transcript", guest_name.lower().replace(" ", "_")],
                    },
                    "usage_permission": "internal_only",
                    "trust_score": "High",
                },
                timeout=TIMEOUT,
            )
        except httpx.TimeoutException:
            return {
                "file": filename,
                "status": "error",
                "reason": "commit timeout",
                "schema": schema,
                "confidence": confidence,
            }
        except Exception as e:
            return {
                "file": filename,
                "status": "error",
                "reason": f"commit error: {e}",
                "schema": schema,
                "confidence": confidence,
            }

        if commit_resp.status_code != 200:
            return {
                "file": filename,
                "status": "error",
                "reason": f"commit {commit_resp.status_code}: {commit_resp.text[:200]}",
                "schema": schema,
                "confidence": confidence,
            }

        commit = commit_resp.json()
        return {
            "file": filename,
            "status": "ok",
            "knowledge_id": commit.get("knowledge_id"),
            "schema": schema,
            "confidence": confidence,
        }


async def main():
    if not TRANSCRIPTS_DIR.exists():
        print(f"Directory not found: {TRANSCRIPTS_DIR}")
        sys.exit(1)

    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    total = len(files)
    print(f"Found {total} transcripts in {TRANSCRIPTS_DIR.name}")
    print(f"API: {API_BASE}")
    print(f"Concurrency: {CONCURRENCY}")
    print("-" * 70)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    success = 0
    errors = 0
    skipped = 0

    async with httpx.AsyncClient() as client:
        # Process in batches for cleaner output
        batch_size = 10
        for batch_start in range(0, total, batch_size):
            batch = files[batch_start : batch_start + batch_size]
            tasks = [ingest_one(client, f, semaphore) for f in batch]
            results = await asyncio.gather(*tasks)

            for result in results:
                status = result["status"]
                if status == "ok":
                    success += 1
                    print(
                        f"  [{success + errors + skipped}/{total}] OK  "
                        f"{result['file']:<45} "
                        f"schema={result['schema']:<15} "
                        f"confidence={result['confidence']}"
                    )
                elif status == "skipped":
                    skipped += 1
                    print(
                        f"  [{success + errors + skipped}/{total}] SKIP "
                        f"{result['file']:<45} "
                        f"{result['reason']}"
                    )
                else:
                    errors += 1
                    print(
                        f"  [{success + errors + skipped}/{total}] ERR  "
                        f"{result['file']:<45} "
                        f"{result['reason'][:60]}"
                    )

    print("-" * 70)
    print(f"Done. {success} ingested, {errors} errors, {skipped} skipped out of {total}")


if __name__ == "__main__":
    asyncio.run(main())
