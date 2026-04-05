# Security findings

## Bandit baseline (2026-04-04)

**HIGH:** 0
**MEDIUM:** 4

| ID | Severity | Description | Location | Status |
|----|----------|-------------|----------|--------|
| B104 | MEDIUM | Binding to all interfaces (0.0.0.0) | ragpipe/app.py:989 | Accepted — container service must bind all interfaces |
| B108 | MEDIUM | Hardcoded temp directory | ragpipe/docstore.py:31 | Accepted — SQLite fallback uses standard temp location |
| B608 | MEDIUM | SQL injection via string query | ragpipe/docstore.py:315 | Accepted — parameterized query with f-string for table name only (not user input) |
| B615 | MEDIUM | HuggingFace Hub download without revision pin | ragpipe/models.py:121 | TODO — pin model revisions for supply chain safety |

## mypy baseline (2026-04-04)

31 errors. Configuration added to pyproject.toml. Type annotations will be improved incrementally.

## Semgrep

Added to CI via security.yml (non-blocking). Scans for OWASP top 10, Python security, and general security audit rules.

## Schemathesis API testing

ragpipe exposes OpenAPI schema at `/openapi.json` (FastAPI default).
TODO: Add schemathesis integration test that starts the app and runs
`st.from_asgi(app).given(...)` for stateful API fuzzing. Blocked on
test infrastructure for the full app lifecycle (requires Qdrant + Postgres).

## OpenSSF Scorecard

Added as a weekly CI workflow. Results published to GitHub Security tab.
