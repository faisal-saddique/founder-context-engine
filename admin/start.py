"""
Admin server launcher for the Founder Context Engine knowledge base.

Reads credentials from the main project .env, wires them into LightRAG Server's
expected env vars, applies the Supabase SSL patch, then starts the server
in-process — no separate install needed.

Usage:
    uv run python admin/start.py

Opens at: http://127.0.0.1:9621
API key:   admin-local
"""
import os
import ssl
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser — skips blanks and comments, strips quotes."""
    env: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            val = raw.split("#")[0].strip().strip('"').strip("'")
            env[key.strip()] = val
    return env


def require(env: dict, *keys: str) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        print(f"[admin] ERROR: Missing in .env: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print(f"[admin] ERROR: .env not found at {env_file}", file=sys.stderr)
        sys.exit(1)

    env = parse_env_file(env_file)
    require(env, "DIRECT_URL", "OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")

    # parse DIRECT_URL → individual PG params
    parsed = urlparse(env["DIRECT_URL"])
    pg = {
        "POSTGRES_HOST":     parsed.hostname or "",
        "POSTGRES_PORT":     str(parsed.port or 5432),
        "POSTGRES_USER":     unquote(parsed.username or ""),
        "POSTGRES_PASSWORD": unquote(parsed.password or ""),
        "POSTGRES_DATABASE": parsed.path.lstrip("/").split("?")[0],
    }

    # set env vars BEFORE any lightrag import so load_dotenv(override=False)
    # in utils_api.py won't clobber them
    os.environ.update({
        # ── server ──────────────────────────────────────────────────────────
        "HOST":                    "127.0.0.1",   # local only — not exposed
        "PORT":                    "9621",
        "WEBUI_TITLE":             "Founder Context Engine — Knowledge Base",
        "WEBUI_DESCRIPTION":       "Inspect the semantic graph and test queries",
        "LIGHTRAG_API_KEY":        "admin-local", # lightweight gate

        # ── LLM: matches lightrag_client.py (gpt_4o_mini_complete) ──────────
        "LLM_BINDING":             "openai",
        "LLM_BINDING_HOST":        "https://api.openai.com/v1",
        "LLM_BINDING_API_KEY":     env["OPENAI_API_KEY"],
        "LLM_MODEL":               "gpt-4o-mini",

        # ── Embedding: MUST match openai_embed defaults ──────────────────────
        # text-embedding-3-small @ 1536d — changing this breaks existing vectors
        "EMBEDDING_BINDING":       "openai",
        "EMBEDDING_BINDING_HOST":  "https://api.openai.com/v1",
        "EMBEDDING_BINDING_API_KEY": env["OPENAI_API_KEY"],
        "EMBEDDING_MODEL":         "text-embedding-3-small",
        "EMBEDDING_DIM":           "1536",

        # ── Storage: mirrors lightrag_client.py exactly ──────────────────────
        "LIGHTRAG_KV_STORAGE":         "PGKVStorage",
        "LIGHTRAG_VECTOR_STORAGE":     "PGVectorStorage",
        "LIGHTRAG_GRAPH_STORAGE":      "Neo4JStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE": "PGDocStatusStorage",

        # ── PostgreSQL (Supabase DIRECT_URL) ─────────────────────────────────
        **pg,
        # require = encrypted TLS, no cert verification (asyncpg semantics)
        "POSTGRES_SSL_MODE":            "require",
        # needed for Supabase's PgBouncer connection pooler
        "POSTGRES_STATEMENT_CACHE_SIZE": "0",
        # conservative pool — read-only admin use
        "POSTGRES_MAX_CONNECTIONS":     "5",

        # ── Neo4j (AuraDB) ───────────────────────────────────────────────────
        "NEO4J_URI":      env["NEO4J_URI"],
        "NEO4J_USERNAME": env["NEO4J_USERNAME"],
        "NEO4J_PASSWORD": env["NEO4J_PASSWORD"],

        # ── Working dir: separate from the production app's rag_storage ──────
        "WORKING_DIR": str(Path(__file__).parent / "rag_storage"),
    })

    # ── Supabase SSL patch ───────────────────────────────────────────────────
    # Supabase uses a private CA with a broken intermediate cert (missing
    # keyUsage). asyncpg's ssl=True uses the default context which fails cert
    # verification. We override _create_ssl_context to use CERT_NONE instead —
    # identical to the fix in src/services/retrieval/lightrag_client.py.
    from lightrag.kg.postgres_impl import PostgreSQLDB

    def _supabase_ssl_context(_self):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    PostgreSQLDB._create_ssl_context = _supabase_ssl_context

    # ── hand off to lightrag-server ──────────────────────────────────────────
    # run from project root so check_env_file() finds the .env there
    os.chdir(str(project_root))

    print("[admin] Starting LightRAG admin server")
    print("[admin] URL:     http://127.0.0.1:9621")
    print("[admin] API key: admin-local  (pass as X-API-Key header)")
    print("[admin] Press Ctrl+C to stop\n")

    from lightrag.api.lightrag_server import main as lightrag_main
    lightrag_main()


if __name__ == "__main__":
    main()
