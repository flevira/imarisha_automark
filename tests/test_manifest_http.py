"""Local fake teacher endpoint exercises the real HTTP client, never a school DB."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from core.manifest_layout import PreparationError
from services.manifest_resolver import CachedManifestResolver, MAX_MANIFEST_BYTES, TeacherManifestClient
from tools.manifest_fixtures import FIXTURES


@pytest.fixture
def server():
    manifest = json.loads((FIXTURES/"personalized-mixed.json").read_text())
    state = {"status":200, "body":json.dumps(manifest).encode(), "calls":[]}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["calls"].append((self.path,self.headers.get("Authorization")))
            if self.headers.get("Authorization") != "Bearer fixture-token":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(state["status"])
            if state["status"] == 302:
                self.send_header("Location", "/untrusted-redirect")
            self.end_headers()
            self.wfile.write(state["body"])
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}", state, manifest
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_authorized_endpoint_path_then_offline_cache(server, tmp_path):
    url, state, manifest = server
    resolver = CachedManifestResolver(tmp_path/"cache.db",url+"|teacher",TeacherManifestClient(url,"fixture-token"))
    assert resolver.resolve("synthetic-mixed-v4","EXAM","42") == manifest
    assert state["calls"] == [("/api/teacher/pdf-exports/layouts/EXAM/42/synthetic-mixed-v4","Bearer fixture-token")]
    resolver.fetch = None
    assert resolver.resolve("synthetic-mixed-v4","EXAM","42") == manifest
    assert len(state["calls"]) == 1
    assert b"fixture-token" not in (tmp_path/"cache.db").read_bytes()


@pytest.mark.parametrize("status,reason", [(403,"access-denied"),(404,"layout-unavailable"),(500,"backend-error"),(302,"backend-error")])
def test_endpoint_failures_and_redirects_do_not_leak_token(server,status,reason):
    url,state,_ = server
    state["status"] = status
    with pytest.raises(PreparationError) as error:
        TeacherManifestClient(url,"fixture-token")("synthetic-mixed-v4","EXAM","42")
    assert error.value.code == reason
    assert len(state["calls"]) == 1
    assert "fixture-token" not in str(error.value)


def test_unauthorized_client_gets_explicit_signin_error(server):
    url,_,_ = server
    with pytest.raises(PreparationError) as error:
        TeacherManifestClient(url,"wrong-token")("synthetic-mixed-v4","EXAM","42")
    assert error.value.code == "authentication-required"


@pytest.mark.parametrize("body", [b"not json", b"[]", b" "*(MAX_MANIFEST_BYTES+1)], ids=["malformed-json", "wrong-root", "oversized"])
def test_invalid_and_oversized_payloads_are_rejected(server,body):
    url,state,_ = server
    state["body"] = body
    with pytest.raises(PreparationError):
        TeacherManifestClient(url,"fixture-token")("synthetic-mixed-v4","EXAM","42")
