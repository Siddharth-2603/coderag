# Source Connectors

Pull external sources into the RAG index by reusing the existing ingestion
pipeline (chunk -> embed -> store). The connector only adds "fetch + clean".

## Confluence connector

`confluence.py` + `ingest_confluence.py`.

### Run (offline demo — no real Confluence)
```bash
../agent/.venv/bin/python ingest_confluence.py --mock --out confluence_index.json
```
Uses canned pages to verify the whole path: fetch -> normalize -> chunk -> embed -> retrieve.

### Run (real Confluence)
Add to `.env` at the repo root:
```
CONFLUENCE_BASE_URL=https://your-org.atlassian.net/wiki
CONFLUENCE_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=xxxx      # Atlassian account -> Security -> API tokens
```
Then drop `--mock`:
```bash
../agent/.venv/bin/python ingest_confluence.py --out confluence_index.json
```
Nothing else changes — the mock and real clients share the same `PageSource` interface.

### Design notes
- **Stable doc id** `CONF:<page_id>` so re-ingesting a page UPDATES it (basis for
  incremental sync) rather than creating duplicates.
- **Metadata for citations**: each chunk carries source/title/space/url/updated_at,
  so answers can link back to the exact Confluence page.
- **HTML normalization**: Confluence returns storage-format XHTML; we strip it to
  text (block tags -> newlines) before chunking.

### Production TODO (describe in a demo, not yet built)
- **ACL / permissions (critical)**: Confluence content is access-controlled. A real
  system MUST enforce per-user permissions at query time — index ACL metadata and
  filter on retrieval, or query the API as the requesting user. Otherwise RAG leaks
  pages a user shouldn't see. Raise this proactively in any company presentation.
- **Incremental sync**: pull only pages changed since last run (CQL `lastmodified`),
  upsert by stable id, delete removed pages. Avoid re-embedding the whole space.
- **Pagination**: the API pages results; a real run follows `_links.next`.
- **Jira**: dynamic data (open tickets, statuses) is better as a LIVE TOOL — a
  `search_jira` tool or the official Atlassian MCP server — not pre-ingested.
