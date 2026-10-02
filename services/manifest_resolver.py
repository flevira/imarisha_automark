"""Account-scoped offline cache and read-only teacher manifest HTTP client."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from core.manifest_layout import PreparationError, require, validate_manifest

MAX_MANIFEST_BYTES = 1_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a teacher token to a redirect destination.


class TeacherManifestClient:
    def __init__(self, base_url: str, token: str, timeout: float = 10):
        parsed = urlsplit(base_url)
        loopback = parsed.hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            pass
        require(parsed.scheme == "https" or (parsed.scheme == "http" and loopback), "Use HTTPS, or HTTP only for a loopback development backend", "invalid-backend")
        require(bool(parsed.hostname) and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
                and parsed.path in ("", "/"), "Backend must be an origin URL", "invalid-backend")
        require(bool(token.strip()) and "\n" not in token and "\r" not in token, "A teacher token is required", "authentication-required")
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout

    def __call__(self, layout_id: str, assessment_type: str, assessment_id: str) -> dict:
        endpoint = "/api/teacher/pdf-exports/layouts/" + "/".join(quote(v, safe="") for v in (assessment_type, assessment_id, layout_id))
        request = Request(self.base_url + endpoint, headers={"Authorization": "Bearer " + self.token, "Accept": "application/json"})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                data = response.read(MAX_MANIFEST_BYTES + 1)
        except HTTPError as exc:
            code = {401: "authentication-required", 403: "access-denied", 404: "layout-unavailable"}.get(exc.code, "backend-error")
            raise PreparationError(code, f"Manifest request returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise PreparationError("backend-unavailable", "Cannot reach the configured manifest backend") from None
        require(len(data) <= MAX_MANIFEST_BYTES, "Manifest response too large")
        try:
            return validate_manifest(json.loads(data))
        except (UnicodeError, json.JSONDecodeError):
            raise PreparationError("invalid-manifest", "Backend returned invalid JSON") from None


class CachedManifestResolver:
    """scope must identify the configured backend AND authenticated staff account.

    Offline access trusts that account's previously authorized cache. Server-side
    revocation cannot be learned offline. refresh=True requires online authorization
    and never falls back to the cache on 401/403 or transport failure.
    """
    def __init__(self, cache_path: Path, scope: str, fetch: Callable[[str, str, str], dict] | None = None, *, refresh: bool = False):
        require(bool(scope.strip()), "An account/backend cache scope is required", "cache-scope-required")
        self.path, self.scope, self.fetch, self.refresh = Path(cache_path), hashlib.sha256(scope.encode()).hexdigest(), fetch, refresh
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS layout_cache(scope TEXT NOT NULL, kind TEXT NOT NULL, assessment_id TEXT NOT NULL, requested_id TEXT NOT NULL, digest TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(scope,kind,assessment_id,requested_id))")
        if os.name != "nt":
            self.path.chmod(0o600)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def _read(self, layout_id: str, kind: str, assessment_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT digest,document FROM layout_cache WHERE scope=? AND kind=? AND assessment_id=? AND requested_id=?", (self.scope, kind, assessment_id, layout_id)).fetchone()
        if row is None:
            return None
        require(hashlib.sha256(row[1].encode()).hexdigest() == row[0], "Cached manifest failed integrity check", "cache-corrupt")
        try:
            return validate_manifest(json.loads(row[1]))
        except (ValueError, TypeError):
            raise PreparationError("cache-corrupt", "Cached manifest is invalid") from None

    def _save(self, requested_id: str, kind: str, assessment_id: str, manifest: dict) -> None:
        document = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        require(len(document.encode()) <= MAX_MANIFEST_BYTES, "Manifest too large")
        digest = hashlib.sha256(document.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest FROM layout_cache WHERE scope=? AND kind=? AND assessment_id=? AND requested_id=?", (self.scope, kind, assessment_id, requested_id)).fetchone()
            require(row is None or row[0] == digest, "Saved immutable layout changed; do not reuse its QR", "layout-conflict")
            db.execute("INSERT OR IGNORE INTO layout_cache VALUES(?,?,?,?,?,?)", (self.scope, kind, assessment_id, requested_id, digest, document))

    def seed_fixture(self, manifest: dict) -> None:
        """Explicit fixture-only import. Production entries come from the authorized endpoint."""
        require(self.scope == hashlib.sha256(b"synthetic-fixtures-only").hexdigest(), "Fixture import requires the isolated fixture scope", "invalid-fixture-scope")
        validate_manifest(manifest)
        self._save(manifest["layoutId"], manifest["assessmentType"], manifest["assessmentId"], manifest)

    def resolve(self, layout_id: str, assessment_type: str, assessment_id: str) -> dict:
        saved = self._read(layout_id, assessment_type, assessment_id)
        if saved is not None and not self.refresh:
            return saved
        if self.fetch is None:
            raise PreparationError("layout-unavailable", "Layout is not available offline; sign in and fetch it first")
        manifest = validate_manifest(self.fetch(layout_id, assessment_type, assessment_id))
        require(manifest["assessmentType"] == assessment_type, "Backend returned a different assessment type", "identity-mismatch")
        # Backend authenticates this exact requested reference, including historical aliases.
        self._save(layout_id, assessment_type, assessment_id, manifest)
        return manifest
