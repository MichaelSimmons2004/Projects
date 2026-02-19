"""
Security pattern library for the static code scanner.

Each pattern dict contains:
  id          : unique snake_case identifier
  category    : human-readable category name
  severity    : "critical" | "high" | "medium" | "low"
  regex       : compiled re.Pattern (case-insensitive where appropriate)
  explanation : what the pattern means and why it matters
  redact      : bool — whether to partially redact the matched text in output
  skip_extensions : set of file extensions where this pattern creates too many FPs
"""
from __future__ import annotations

import re

# Shorthand
_I = re.IGNORECASE


PATTERNS: list[dict] = [

    # ═══════════════════════════════════════════════════════════════════════
    # HARDCODED CREDENTIALS
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "hardcoded_password",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(
            r'(?:password|passwd|pwd|secret|pass)\s*[=:]\s*["\'](?!.*\{)[^\s"\']{6,}["\']',
            _I,
        ),
        "explanation": (
            "A hardcoded password or secret was found. "
            "Credentials should be loaded from environment variables or a secrets manager, never embedded in code."
        ),
        "redact": True,
    },
    {
        "id": "hardcoded_api_key",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(
            r'(?:api[_-]?key|apikey|access[_-]?key|auth[_-]?token|bearer[_-]?token)\s*[=:]\s*["\'][A-Za-z0-9\-_]{16,}["\']',
            _I,
        ),
        "explanation": (
            "A hardcoded API key or auth token was found. "
            "Rotate this key immediately and move it to environment variables."
        ),
        "redact": True,
    },
    {
        "id": "aws_access_key",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(r"(?<![A-Z0-9])(AKIA|AGPA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
        "explanation": (
            "An AWS access key ID was detected. "
            "If this key is active, rotate it immediately in the AWS IAM console."
        ),
        "redact": True,
    },
    {
        "id": "aws_secret_key",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(
            r'(?:aws[_-]?secret[_-]?access[_-]?key|aws[_-]?secret)\s*[=:]\s*["\'][A-Za-z0-9/+=]{40}["\']',
            _I,
        ),
        "explanation": "An AWS secret access key was detected in source code.",
        "redact": True,
    },
    {
        "id": "private_key_header",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(r"-----BEGIN (RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
        "explanation": (
            "A private key header was found in source code. "
            "Private keys must never be committed to version control."
        ),
        "redact": False,
    },
    {
        "id": "github_token",
        "category": "Hardcoded Credential",
        "severity": "critical",
        "regex": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
        "explanation": "A GitHub personal access token or OAuth token was detected.",
        "redact": True,
    },
    {
        "id": "slack_token",
        "category": "Hardcoded Credential",
        "severity": "high",
        "regex": re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
        "explanation": "A Slack API token was detected.",
        "redact": True,
    },
    {
        "id": "jwt_token",
        "category": "Hardcoded Credential",
        "severity": "high",
        "regex": re.compile(r"eyJ[A-Za-z0-9+/=]{20,}\.[A-Za-z0-9+/=]{20,}\.[A-Za-z0-9+/=\-_]{20,}"),
        "explanation": (
            "A JWT token was detected hardcoded in source. "
            "JWTs often carry authentication claims and should not be embedded in code."
        ),
        "redact": True,
    },
    {
        "id": "generic_secret_assignment",
        "category": "Hardcoded Credential",
        "severity": "medium",
        "regex": re.compile(
            r'(?:token|secret|credential|private_key|auth)\s*=\s*["\'][A-Za-z0-9+/=\-_]{20,}["\']',
            _I,
        ),
        "explanation": (
            "A generic secret or token assignment was found. "
            "Verify this is not a real credential before committing."
        ),
        "redact": True,
        "skip_extensions": {".md", ".txt", ".rst"},
    },

    # ═══════════════════════════════════════════════════════════════════════
    # COMMAND INJECTION
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "os_system_call",
        "category": "Command Injection",
        "severity": "high",
        "regex": re.compile(r"\bos\.system\s*\("),
        "explanation": (
            "os.system() is dangerous when its argument contains any user-controlled data. "
            "Prefer subprocess.run() with a list of arguments and shell=False."
        ),
        "redact": False,
        "skip_extensions": {".md"},
    },
    {
        "id": "subprocess_shell_true",
        "category": "Command Injection",
        "severity": "high",
        "regex": re.compile(r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True", _I),
        "explanation": (
            "subprocess with shell=True passes the command to the shell interpreter. "
            "If any part of the command string is user-controlled, this enables command injection."
        ),
        "redact": False,
    },
    {
        "id": "eval_exec_call",
        "category": "Command Injection",
        "severity": "critical",
        "regex": re.compile(r"\b(?:eval|exec)\s*\((?!\s*[\"\']{3})"),
        "explanation": (
            "eval() and exec() execute arbitrary Python code. "
            "Never pass user-supplied or external data to these functions."
        ),
        "redact": False,
        "skip_extensions": {".md", ".txt"},
    },
    {
        "id": "php_system_exec",
        "category": "Command Injection",
        "severity": "critical",
        "regex": re.compile(r"\b(?:system|exec|shell_exec|passthru|popen)\s*\(\s*\$", _I),
        "explanation": (
            "PHP shell execution function called with a variable argument. "
            "This is a classic command injection vector if the variable contains user input."
        ),
        "redact": False,
    },
    {
        "id": "node_child_process_exec",
        "category": "Command Injection",
        "severity": "high",
        "regex": re.compile(r"(?:exec|execSync|spawn|spawnSync)\s*\(\s*[^,)]*\+", _I),
        "explanation": (
            "String concatenation in a Node.js child_process call. "
            "If the concatenated value includes user input, this enables command injection."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SQL INJECTION
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "sql_string_format",
        "category": "SQL Injection",
        "severity": "critical",
        "regex": re.compile(
            r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER)\b.{0,80}"
            r"(?:%s|%d|\.format\s*\(|\+\s*[a-z_]\w*|f[\"'].*\{)",
            _I,
        ),
        "explanation": (
            "SQL query constructed with string formatting or concatenation. "
            "Use parameterised queries or an ORM to prevent SQL injection."
        ),
        "redact": False,
    },
    {
        "id": "raw_sql_execute",
        "category": "SQL Injection",
        "severity": "high",
        "regex": re.compile(r"\.execute\s*\(\s*[\"'].*(?:SELECT|INSERT|UPDATE|DELETE)", _I),
        "explanation": (
            "Raw SQL passed directly to .execute(). "
            "Ensure this query never incorporates user input without parameterisation."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-SITE SCRIPTING (XSS)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "jinja2_autoescape_off",
        "category": "XSS",
        "severity": "high",
        "regex": re.compile(r"autoescape\s*=\s*False", _I),
        "explanation": (
            "Jinja2 autoescaping is disabled. "
            "All template variables will be rendered as raw HTML, enabling XSS if they contain user data."
        ),
        "redact": False,
    },
    {
        "id": "innerHTML_assignment",
        "category": "XSS",
        "severity": "high",
        "regex": re.compile(r"\.innerHTML\s*=(?!=)", _I),
        "explanation": (
            "Direct innerHTML assignment. "
            "If the value contains user-supplied data, this enables stored or reflected XSS."
        ),
        "redact": False,
        "skip_extensions": {".md"},
    },
    {
        "id": "document_write",
        "category": "XSS",
        "severity": "medium",
        "regex": re.compile(r"document\.write\s*\(", _I),
        "explanation": (
            "document.write() with dynamic content can introduce XSS. "
            "Use textContent or createElement instead."
        ),
        "redact": False,
    },
    {
        "id": "dangerously_set_inner_html",
        "category": "XSS",
        "severity": "high",
        "regex": re.compile(r"dangerouslySetInnerHTML\s*=", _I),
        "explanation": (
            "React's dangerouslySetInnerHTML bypasses XSS protection. "
            "Ensure the value is sanitised with a library like DOMPurify."
        ),
        "redact": False,
    },
    {
        "id": "php_echo_get_post",
        "category": "XSS",
        "severity": "critical",
        "regex": re.compile(r"echo\s+\$_(?:GET|POST|REQUEST|COOKIE)\s*\[", _I),
        "explanation": (
            "PHP echo of unfiltered $_GET/$_POST/$_REQUEST/$_COOKIE data. "
            "Always call htmlspecialchars() before echoing user input."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # REVERSE SHELLS
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "bash_reverse_shell",
        "category": "Reverse Shell",
        "severity": "critical",
        "regex": re.compile(
            r"bash\s+-i\s+>&?\s*/dev/tcp/|"
            r"bash\s+-c\s+['\"].*\b(?:0>&1|>&\s*/dev/tcp)",
            _I,
        ),
        "explanation": (
            "A bash reverse shell pattern was detected. "
            "This is a classic technique used to establish a remote shell connection back to an attacker."
        ),
        "redact": False,
    },
    {
        "id": "python_reverse_shell",
        "category": "Reverse Shell",
        "severity": "critical",
        "regex": re.compile(
            r"import\s+socket.*\bconnect\b|"
            r"socket\.socket\(\).*\bconnect\b.*subprocess|"
            r"pty\.spawn\s*\(\s*['\"](?:/bin/sh|/bin/bash|cmd)['\"]",
            _I,
        ),
        "explanation": (
            "A Python reverse shell pattern was detected. "
            "This code connects a socket to a remote host and spawns an interactive shell."
        ),
        "redact": False,
    },
    {
        "id": "nc_reverse_shell",
        "category": "Reverse Shell",
        "severity": "critical",
        "regex": re.compile(r"\bnc(?:at)?\b.*-e\s+(?:/bin/|cmd)", _I),
        "explanation": (
            "netcat with -e flag detected — this is used to bind a shell to a network connection, "
            "a core reverse/bind shell technique."
        ),
        "redact": False,
    },
    {
        "id": "powershell_download_execute",
        "category": "Reverse Shell",
        "severity": "critical",
        "regex": re.compile(
            r"(?:IEX|Invoke-Expression)\s*\(.*(?:Net\.WebClient|Invoke-WebRequest|iwr\b)|"
            r"powershell\s+.*-(?:enc|encodedcommand)\s+[A-Za-z0-9+/=]{20,}",
            _I,
        ),
        "explanation": (
            "PowerShell download-and-execute or encoded command pattern detected. "
            "Commonly used in malware droppers, living-off-the-land attacks, and staged payloads."
        ),
        "redact": False,
    },
    {
        "id": "msfvenom_payload_indicator",
        "category": "Reverse Shell",
        "severity": "critical",
        "regex": re.compile(
            r"msfvenom|meterpreter|metasploit|"
            r"\bpayload\b.*(?:windows/|linux/|osx/).*(?:reverse|bind)_",
            _I,
        ),
        "explanation": (
            "Metasploit/msfvenom payload reference detected. "
            "Ensure this is intentional (e.g. legitimate penetration testing) and not accidental exposure."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SUSPICIOUS ENCODING / OBFUSCATION
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "large_base64_blob",
        "category": "Suspicious Encoding",
        "severity": "medium",
        "regex": re.compile(r"['\"]([A-Za-z0-9+/]{200,}={0,2})['\"]"),
        "explanation": (
            "A large base64-encoded blob was detected. "
            "This may be an obfuscated payload, shellcode, or embedded binary. Decode and inspect."
        ),
        "redact": True,
        "skip_extensions": {".md", ".txt", ".rst", ".json"},
    },
    {
        "id": "hex_shellcode",
        "category": "Suspicious Encoding",
        "severity": "high",
        "regex": re.compile(r"(?:\\x[0-9a-f]{2}){20,}", _I),
        "explanation": (
            "A long sequence of hex-escaped bytes was detected. "
            "This pattern is commonly used to embed shellcode or obfuscated payloads in code."
        ),
        "redact": True,
    },
    {
        "id": "obfuscated_js",
        "category": "Suspicious Encoding",
        "severity": "medium",
        "regex": re.compile(
            r"(?:eval|Function)\s*\(\s*(?:atob|unescape|decodeURIComponent)\s*\(",
            _I,
        ),
        "explanation": (
            "Obfuscated JavaScript using eval() with a decoding function. "
            "This is a classic technique to hide malicious scripts from static analysis."
        ),
        "redact": False,
        "skip_extensions": {".py", ".rb", ".java", ".go"},
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DANGEROUS IMPORTS / DESERIALISATION
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "python_pickle_load",
        "category": "Dangerous Deserialisation",
        "severity": "high",
        "regex": re.compile(r"pickle\.(?:load|loads|Unpickler)\s*\("),
        "explanation": (
            "pickle.load() deserialises arbitrary Python objects. "
            "Never deserialise data from untrusted sources — this can lead to RCE."
        ),
        "redact": False,
    },
    {
        "id": "python_marshal_loads",
        "category": "Dangerous Deserialisation",
        "severity": "high",
        "regex": re.compile(r"marshal\.loads\s*\("),
        "explanation": (
            "marshal.loads() deserialises Python bytecode. "
            "Loading untrusted marshalled data can execute arbitrary code."
        ),
        "redact": False,
    },
    {
        "id": "yaml_unsafe_load",
        "category": "Dangerous Deserialisation",
        "severity": "high",
        "regex": re.compile(r"yaml\.(?:load|Loader)\s*\([^,)]*(?!\bSafe)", _I),
        "explanation": (
            "yaml.load() without SafeLoader can deserialise arbitrary Python objects. "
            "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader) instead."
        ),
        "redact": False,
    },
    {
        "id": "xml_external_entity",
        "category": "XXE / XML",
        "severity": "high",
        "regex": re.compile(r"xml\.etree\.ElementTree|lxml\.etree|minidom\.parseString", _I),
        "explanation": (
            "XML parsing detected. Verify that external entity processing (XXE) is disabled, "
            "especially if parsing untrusted XML input. Use defusedxml for safe parsing."
        ),
        "redact": False,
        "skip_extensions": {".md", ".txt"},
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SSRF / OPEN REDIRECT
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "requests_user_controlled_url",
        "category": "SSRF",
        "severity": "high",
        "regex": re.compile(
            r"requests\.\w+\s*\(\s*(?:url|request\.(?:args|form|json|data|get|POST))",
            _I,
        ),
        "explanation": (
            "A requests call appears to use a user-controlled URL. "
            "Without allowlist validation, this enables Server-Side Request Forgery (SSRF)."
        ),
        "redact": False,
    },
    {
        "id": "open_redirect",
        "category": "Open Redirect",
        "severity": "medium",
        "regex": re.compile(
            r"(?:redirect|location)\s*\(\s*request\.(?:args|form|get|params)",
            _I,
        ),
        "explanation": (
            "An HTTP redirect using user-supplied data may allow open redirect attacks. "
            "Validate the destination URL against an allowlist."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PATH TRAVERSAL
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "path_traversal_pattern",
        "category": "Path Traversal",
        "severity": "high",
        "regex": re.compile(r"\.\./|\.\.\\\\"),
        "explanation": (
            "Directory traversal sequence (../) detected. "
            "If this value originates from user input, it may allow reading files outside the intended directory."
        ),
        "redact": False,
        "skip_extensions": {".md", ".txt", ".rst"},
    },
    {
        "id": "open_with_user_path",
        "category": "Path Traversal",
        "severity": "medium",
        "regex": re.compile(r"\bopen\s*\(\s*(?:request|req|params|args|user)\b", _I),
        "explanation": (
            "File open() call with what appears to be a user-controlled path variable. "
            "Sanitise and validate the path before passing it to open()."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CRYPTOGRAPHY WEAKNESSES
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "weak_hash_md5_sha1",
        "category": "Weak Cryptography",
        "severity": "medium",
        "regex": re.compile(r"\b(?:md5|sha1|MD5|SHA1)\b.*(?:hash|digest|encrypt|password)", _I),
        "explanation": (
            "MD5 or SHA-1 used in a security-sensitive context. "
            "These algorithms are cryptographically broken; use SHA-256 or bcrypt/argon2 for passwords."
        ),
        "redact": False,
        "skip_extensions": {".md", ".txt"},
    },
    {
        "id": "hardcoded_iv_key",
        "category": "Weak Cryptography",
        "severity": "high",
        "regex": re.compile(
            r"(?:iv|nonce|key)\s*=\s*b?[\"'][\\x0-9a-fA-F]{8,}[\"']",
            _I,
        ),
        "explanation": (
            "A hardcoded IV, nonce, or encryption key was detected. "
            "IVs/nonces should be randomly generated per operation; keys should be loaded from a secrets manager."
        ),
        "redact": True,
    },
    {
        "id": "ssl_verify_false",
        "category": "Weak Cryptography",
        "severity": "high",
        "regex": re.compile(r"verify\s*=\s*False", _I),
        "explanation": (
            "TLS/SSL certificate verification is disabled. "
            "This allows man-in-the-middle attacks. Never disable verification in production."
        ),
        "redact": False,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # DEBUG / INFORMATION EXPOSURE
    # ═══════════════════════════════════════════════════════════════════════

    {
        "id": "flask_debug_true",
        "category": "Debug/Info Exposure",
        "severity": "high",
        "regex": re.compile(r"app\.run\s*\([^)]*debug\s*=\s*True", _I),
        "explanation": (
            "Flask debug mode is enabled. "
            "The Werkzeug debugger exposes an interactive Python console — never use in production."
        ),
        "redact": False,
    },
    {
        "id": "stack_trace_exposure",
        "category": "Debug/Info Exposure",
        "severity": "medium",
        "regex": re.compile(r"traceback\.print_exc\(\)|traceback\.format_exc\(\)", _I),
        "explanation": (
            "Stack traces may be exposed to end users. "
            "Log them server-side and return a generic error message to the client."
        ),
        "redact": False,
    },
    {
        "id": "todo_security_note",
        "category": "Debug/Info Exposure",
        "severity": "low",
        "regex": re.compile(r"#\s*TODO\s*:?\s*(?:fix|secure|sanitize|auth|validate)", _I),
        "explanation": (
            "A TODO comment mentioning a security concern was found. "
            "Ensure this is tracked and addressed before deploying."
        ),
        "redact": False,
    },
]
