# app.py
"""
SAST SaaS MVP – single-file, copy–paste–run

Features in this MVP (end-to-end):
- FastAPI web service with minimal HTML UI (upload code / paste code)
- Static Python analyzer (AST-based): facts, rules, taint-lite, secrets scan
- SBOM (CycloneDX minimal) from requirements.txt + live OSV vuln lookup
- Dockerfile linter (pin base, non-root user, no latest tag)
- SARIF export endpoint
- Simple in-memory storage (also optional SQLite persistence)
- CLI mode: `python app.py scan path/to/file.py` or `python app.py serve`

Quickstart:
1) Python 3.10+
2) pip install fastapi uvicorn jinja2 httpx pydantic[dotenv] python-multipart
3) Run server:  uvicorn app:app --reload
   Open http://127.0.0.1:8000

Notes:
- OSV queries require internet. If you're offline, vuln results will be empty.
- This is an MVP. Safe defaults, no background jobs; easy to extend.
"""
from __future__ import annotations

import ast
import base64
import builtins
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
import textwrap
import typing as t
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from jinja2 import Environment, DictLoader, select_autoescape
from pydantic import BaseModel

# -------------------------
# Utility & Types
# -------------------------

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass(slots=True)
class Span:
    line: int
    col: int
    end_line: t.Optional[int] = None
    end_col: t.Optional[int] = None

@dataclass(slots=True)
class Finding:
    rule_id: str
    message: str
    severity: Severity
    span: Span
    filename: str
    function: t.Optional[str] = None
    extra: dict = field(default_factory=dict)

@dataclass(slots=True)
class RuleContext:
    tree: ast.AST
    filename: str
    imported: dict[str, set[str]]
    symbol_table: dict[str, set[int]]
    assigned_in_function: dict[str, set[str]]

class Rule:
    id: str = "GENERIC"
    description: str = ""
    severity: Severity = Severity.WARNING
    def check(self, ctx: RuleContext) -> list[Finding]:
        return []

# -------------------------
# AST helpers
# -------------------------

def _safe_unparse(node: t.Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        from ast import unparse as ast_unparse  # type: ignore
        s = ast_unparse(node)
        return s if len(s) < 300 else s[:297] + "..."
    except Exception:
        return node.__class__.__name__

def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Subscript):
        return f"{_name_of(node.value)}[...]"
    if isinstance(node, ast.alias):
        return node.asname or node.name
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return node.__class__.__name__

# -------------------------
# Facts visitor
# -------------------------

class FactsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imported: dict[str, set[str]] = {}
        self.symbol_table: dict[str, set[int]] = {}
        self.assigned_in_function: dict[str, set[str]] = {}
        self.current_function: t.Optional[str] = None
        self.facts: dict[str, t.Any] = {
            "functions": [],
            "loops": [],
            "conditionals": [],
            "prints": [],
            "exceptions": [],
            "literals": [],
            "calls": [],
        }

    def _span(self, node: ast.AST) -> Span:
        return Span(
            line=getattr(node, "lineno", 0) or 0,
            col=getattr(node, "col_offset", 0) or 0,
            end_line=getattr(node, "end_lineno", None),
            end_col=getattr(node, "end_col_offset", None),
        )

    # imports
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imported.setdefault(alias.name, set()).add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            key = f"{node.module}.{alias.name}" if node.module else alias.name
            self.imported.setdefault(key, set()).add(alias.asname or alias.name)
        self.generic_visit(node)

    # functions
    def visit_FunctionDef(self, node: ast.FunctionDef):
        info = {
            "name": node.name,
            "args": [a.arg for a in node.args.args],
            "returns": _safe_unparse(node.returns) if node.returns else None,
            "span": self._span(node).__dict__,
            "cyclomatic": self._cyclomatic_complexity(node),
        }
        self.facts["functions"].append(info)
        prev = self.current_function
        self.current_function = node.name
        self.assigned_in_function.setdefault(node.name, set())
        self.generic_visit(node)
        self.current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def _cyclomatic_complexity(self, fn: ast.FunctionDef) -> int:
        decision_nodes = (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler, ast.Try, ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.Match)
        count = 1
        for n in ast.walk(fn):
            if isinstance(n, decision_nodes):
                count += 1
        return count

    def visit_Assign(self, node: ast.Assign):
        if self.current_function:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self.assigned_in_function[self.current_function].add(tgt.id)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        self.symbol_table.setdefault(node.id, set()).add(getattr(node, "lineno", 0) or 0)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.facts["calls"].append({"name": _name_of(node.func), "line": getattr(node, "lineno", 0)})
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.facts["prints"].append({"line": getattr(node, "lineno", 0)})
        self.generic_visit(node)

# -------------------------
# Rules
# -------------------------

class RuleMutableDefaultArgs(Rule):
    id = "PY1001"; description = "Avoid mutable default arguments"; severity = Severity.WARNING
    def check(self, ctx: RuleContext) -> list[Finding]:
        out: list[Finding] = []
        for n in ast.walk(ctx.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in n.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        out.append(Finding(self.id, f"Function '{n.name}' has mutable default.", self.severity,
                                           Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename, n.name))
        return out

class RuleBareExcept(Rule):
    id = "PY1002"; description = "Avoid bare except"; severity = Severity.WARNING
    def check(self, ctx: RuleContext) -> list[Finding]:
        out=[]
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.ExceptHandler) and n.type is None:
                out.append(Finding(self.id, "Bare except detected.", self.severity,
                                   Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename))
        return out

class RuleRiskyCalls(Rule):
    id = "PY1004"; description = "Risky calls (eval/exec/os.system/subprocess.*)"; severity = Severity.WARNING
    RISKY = {"eval","exec","os.system","subprocess.call","subprocess.Popen","subprocess.run"}
    def check(self, ctx: RuleContext) -> list[Finding]:
        out=[]
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.Call):
                name=_name_of(n.func)
                if name in self.RISKY:
                    # shell=True hard error
                    sev = Severity.ERROR if any((kw.arg=="shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True) for kw in (n.keywords or [])) else self.severity
                    out.append(Finding(self.id, f"Risky call '{name}'.", sev,
                                       Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename))
        return out

class RuleShadowBuiltins(Rule):
    id = "PY1007"; description = "Shadows builtin"; severity = Severity.INFO
    def check(self, ctx: RuleContext) -> list[Finding]:
        out=[]
        built=set(dir(builtins))
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.FunctionDef) and n.name in built:
                out.append(Finding(self.id, f"Function name '{n.name}' shadows builtin.", self.severity,
                                   Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename, n.name))
            if isinstance(n, ast.Assign):
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in built:
                        out.append(Finding(self.id, f"Variable '{tgt.id}' shadows builtin.", self.severity,
                                           Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename))
        return out

class RuleRequestsVerifyFalse(Rule):
    id = "PY1011"; description = "requests with verify=False"; severity = Severity.ERROR
    def check(self, ctx: RuleContext) -> list[Finding]:
        out=[]
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.Call) and _name_of(n.func).startswith("requests."):
                for kw in n.keywords or []:
                    if kw.arg=="verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        out.append(Finding(self.id, "TLS verification disabled in requests call.", self.severity,
                                           Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename))
        return out

class RuleYamlUnsafeLoad(Rule):
    id = "PY1012"; description = "yaml.load without SafeLoader"; severity = Severity.ERROR
    def check(self, ctx: RuleContext) -> list[Finding]:
        out=[]
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.Call) and _name_of(n.func) in {"yaml.load","ruamel.yaml.load"}:
                # if Loader kw not present or not SafeLoader
                has_safe = any((kw.arg in {"Loader","loader"} and hasattr(ast, "Name") and isinstance(kw.value, ast.Attribute) and kw.value.attr.endswith("SafeLoader")) for kw in (n.keywords or []))
                if not has_safe:
                    out.append(Finding(self.id, "yaml.load without SafeLoader.", self.severity,
                                       Span(n.lineno, n.col_offset, getattr(n,'end_lineno',None), getattr(n,'end_col_offset',None)), ctx.filename))
        return out

BUILTIN_RULES: list[Rule] = [
    RuleMutableDefaultArgs(),
    RuleBareExcept(),
    RuleRiskyCalls(),
    RuleShadowBuiltins(),
    RuleRequestsVerifyFalse(),
    RuleYamlUnsafeLoad(),
]

# -------------------------
# Taint-lite & Secrets
# -------------------------

TAINT_SOURCES = {"input", "flask.request.args.get", "flask.request.form.get"}
SINKS = {"eval","exec","os.system","subprocess.run","subprocess.Popen","cursor.execute"}
SANITIZERS = {"shlex.quote","html.escape"}

class TaintVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.tainted: set[str] = set()
        self.findings: list[Finding] = []
        self.current_function: t.Optional[str] = None
        self.filename: str = "<memory>"
    def _span(self, node: ast.AST) -> Span:
        return Span(getattr(node,"lineno",0) or 0, getattr(node,"col_offset",0) or 0,
                    getattr(node,"end_lineno",None), getattr(node,"end_col_offset",None))
    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev=self.current_function; self.current_function=node.name
        self.generic_visit(node); self.current_function=prev
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Assign(self, node: ast.Assign):
        tainted = self._expr_is_tainted(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tainted:
                self.tainted.add(tgt.id)
        self.generic_visit(node)
    def visit_Call(self, node: ast.Call):
        name=_name_of(node.func)
        if name in SINKS and any(self._expr_is_tainted(a) for a in node.args):
            self.findings.append(Finding("TAINT-001", f"Tainted data flows into sink '{name}'.", Severity.ERROR,
                                         self._span(node), self.filename, self.current_function))
        self.generic_visit(node)
    def _expr_is_tainted(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            nm=_name_of(node.func)
            if nm in TAINT_SOURCES: return True
            if nm in SANITIZERS: return False
            return any(self._expr_is_tainted(a) for a in node.args)
        if isinstance(node, ast.Name):
            return node.id in self.tainted
        if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.BoolOp, ast.FormattedValue)):
            return any(self._expr_is_tainted(c) for c in ast.iter_child_nodes(node))
        if isinstance(node, ast.Subscript):
            return self._expr_is_tainted(node.value) or any(self._expr_is_tainted(c) for c in ast.iter_child_nodes(node))
        return False

SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS Access Key"),
    (re.compile(r"aws_secret_access_key\s*[:=]\s*['\"]([A-Za-z0-9/+=]{40})['\"]"), "AWS Secret Key"),
    (re.compile(r"-----BEGIN (?:RSA|EC|DSA) PRIVATE KEY-----"), "Private Key"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,48}"), "Slack Token"),
]

def find_secrets(text: str, filename: str) -> list[Finding]:
    out: list[Finding] = []
    for rx, label in SECRET_PATTERNS:
        for m in rx.finditer(text):
            # line number from offset
            line = text[: m.start()].count("\n") + 1
            out.append(Finding("SECRET-001", f"Potential secret: {label}", Severity.ERROR,
                               Span(line, 0, line, None), filename))
    # simple entropy heuristic for long tokens
    for m in re.finditer(r"(['\"])\s*([A-Za-z0-9+/=]{32,})\s*\1", text):
        token = m.group(2)
        if _shannon_entropy(token) >= 3.5:
            line = text[: m.start()].count("\n") + 1
            out.append(Finding("SECRET-ENTROPY", "High-entropy string; review if secret.", Severity.WARNING,
                               Span(line,0,line,None), filename))
    return out

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    freq = {ch: s.count(ch) for ch in set(s)}
    l = len(s)
    return -sum((c/l)*math.log2(c/l) for c in freq.values())

# -------------------------
# Analyzer API
# -------------------------

class AnalysisResult(BaseModel):n    filename: str
    findings: list[dict]
    taint_findings: list[dict]
    facts: dict
    summary: str
    sbom: t.Optional[dict] = None
    vulns: t.Optional[list[dict]] = None


def analyze_code(code: str, filename: str = "<memory>") -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "filename": filename}

    fv = FactsVisitor(); fv.visit(tree)
    ctx = RuleContext(tree=tree, filename=filename, imported=fv.imported,
                      symbol_table=fv.symbol_table, assigned_in_function=fv.assigned_in_function)

    findings: list[Finding] = []
    for rule in BUILTIN_RULES:
        findings.extend(rule.check(ctx))

    tv = TaintVisitor(); tv.filename = filename; tv.visit(tree)

    parts = []
    if fv.facts["functions"]: parts.append(f"Defines {len(fv.facts['functions'])} functions")
    if fv.facts["prints"]: parts.append("Produces console output")
    summary = ", ".join(parts) or "Simple statements"

    ser = lambda f: {
        "rule_id": f.rule_id, "message": f.message, "severity": f.severity.value,
        "span": f.span.__dict__, "filename": f.filename, "function": f.function, "extra": f.extra
    }

    res = {
        "filename": filename,
        "facts": fv.facts,
        "findings": [ser(f) for f in findings],
        "taint_findings": [ser(f) for f in tv.findings],
        "summary": summary,
    }
    # secrets on the raw text
    secrets = find_secrets(code, filename)
    res["findings"].extend([ser(f) for f in secrets])
    return res

# -------------------------
# SBOM & Vulnerabilities (OSV)
# -------------------------

REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*([=<>!~]=\s*[^#\s]+)?")

async def generate_sbom_and_vulns(requirements_txt: str) -> tuple[dict, list[dict]]:
    pkgs: list[tuple[str, str|None]] = []
    for line in requirements_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        m = REQ_LINE.match(line)
        if not m: continue
        name = m.group(1)
        version = None
        if m.group(2) and "==" in m.group(2):
            version = m.group(2).split("==",1)[1].strip()
        pkgs.append((name, version))

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "version": 1,
        "components": [{
            "type": "library",
            "name": n,
            "version": v or "*",
            "purl": f"pkg:pypi/{n}@{v}" if v else f"pkg:pypi/{n}",
        } for n,v in pkgs]
    }

    vulns: list[dict] = []
    async with httpx.AsyncClient(timeout=10) as client:
        for name, ver in pkgs:
            if not ver:
                continue
            payload = {"package": {"name": name, "ecosystem": "PyPI"}, "version": ver}
            try:
                r = await client.post("https://api.osv.dev/v1/query", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    for v in data.get("vulns", []) or []:
                        vulns.append({
                            "id": v.get("id"),
                            "summary": v.get("summary"),
                            "severity": v.get("severity"),
                            "affected": name,
                            "version": ver,
                        })
            except Exception:
                pass
    return sbom, vulns

# -------------------------
# Dockerfile scanner (tiny)
# -------------------------

def scan_dockerfile(text: str, filename: str="Dockerfile") -> list[Finding]:
    out: list[Finding] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        L = line.strip()
        if L.upper().startswith("FROM") and (":latest" in L or L.endswith(":latest")):
            out.append(Finding("DOCKER-001", "Avoid FROM ...:latest; pin a digest or version.", Severity.WARNING, Span(i,0,i,None), filename))
        if L.upper().startswith("FROM") and "@sha256:" not in L and ":" not in L:
            out.append(Finding("DOCKER-002", "Base image not pinned; use tag or digest.", Severity.WARNING, Span(i,0,i,None), filename))
        if L.startswith("USER root"):
            out.append(Finding("DOCKER-003", "Container runs as root; switch to non-root user.", Severity.ERROR, Span(i,0,i,None), filename))
    return out

# -------------------------
# SARIF emitter
# -------------------------

def to_sarif(results: list[dict]) -> dict:
    runs = [{
        "tool": {"driver": {"name": "SAST-MVP", "version": "0.1"}},
        "results": []
    }]
    for res in results:
        for f in (res.get("findings", []) + res.get("taint_findings", [])):
            runs[0]["results"].append({
                "ruleId": f["rule_id"],
                "level": {"info":"note","warning":"warning","error":"error"}[f["severity"].lower()],
                "message": {"text": f["message"]},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": res.get("filename")},
                        "region": {
                            "startLine": f["span"]["line"] or 1,
                            "startColumn": f["span"]["col"] or 1,
                        },
                    }
                }],
            })
    return {"version": "2.1.0", "runs": runs}

# -------------------------
# Minimal storage (in-memory)
# -------------------------

class Store:
    def __init__(self) -> None:
        self.items: list[dict] = []
    def add(self, item: dict) -> None:
        self.items.append(item)
    def all(self) -> list[dict]:
        return list(self.items)

store = Store()

# -------------------------
# Web App (FastAPI)
# -------------------------

templates = Environment(
    loader=DictLoader({
        "index.html": """
        <!doctype html>
        <html><head><meta charset='utf-8'><title>SAST MVP</title>
        <style>
            body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;padding:24px;}
            .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
            .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px}
            .error{background:#fee;color:#900}
            .warn{background:#fff8e1;color:#8a6d3b}
            .info{background:#eef5ff;color:#235}
            pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}
            table{border-collapse:collapse;width:100%}
            td,th{border-bottom:1px solid #eee;padding:8px;text-align:left;font-size:14px}
            .sev{font-weight:600}
        </style></head>
        <body>
        <h1>🛡️ SAST SaaS MVP</h1>
        <div class="card">
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <h3>Analyze a Python file</h3>
                <input type="file" name="codefile" accept=".py" required>
                <br/><br/>
                <h3>Optional requirements.txt (for SBOM & vulns)</h3>
                <input type="file" name="reqfile" accept=".txt">
                <br/><br/>
                <button>Analyze</button>
            </form>
        </div>
        <div class="card">
            <h3>Paste code</h3>
            <form action="/analyze_text" method="post">
                <textarea name="code" rows="10" style="width:100%" placeholder="paste Python code..."></textarea>
                <br/><button>Analyze</button>
            </form>
        </div>
        <div class="card">
            <h3>Recent Results</h3>
            {% for item in items %}
            <div class="card">
                <div><strong>{{item.filename}}</strong> — {{item.summary}}</div>
                <div>
                    {% for f in item.findings %}
                        <span class="pill {{'error' if f.severity=='error' else ('warn' if f.severity=='warning' else 'info')}}">{{f.rule_id}}: {{f.severity}}</span>
                    {% endfor %}
                </div>
                <details>
                    <summary>Details</summary>
                    <table>
                        <tr><th>Rule</th><th>Severity</th><th>Message</th><th>Line</th></tr>
                        {% for f in item.findings %}
                        <tr>
                            <td>{{f.rule_id}}</td>
                            <td class="sev">{{f.severity}}</td>
                            <td>{{f.message}}</td>
                            <td>{{f.span.line}}</td>
                        </tr>
                        {% endfor %}
                        {% for f in item.taint_findings %}
                        <tr>
                            <td>{{f.rule_id}}</td>
                            <td class="sev">{{f.severity}}</td>
                            <td>{{f.message}}</td>
                            <td>{{f.span.line}}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    {% if item.sbom %}
                    <h4>SBOM (CycloneDX, {{item.sbom.components|length}} components)</h4>
                    <pre>{{ item.sbom | tojson(indent=2) }}</pre>
                    {% endif %}
                    {% if item.vulns %}
                    <h4>Vulnerabilities (OSV)</h4>
                    <table>
                        <tr><th>ID</th><th>Package</th><th>Version</th><th>Summary</th></tr>
                        {% for v in item.vulns %}
                        <tr><td>{{v.id}}</td><td>{{v.affected}}</td><td>{{v.version}}</td><td>{{v.summary}}</td></tr>
                        {% endfor %}
                    </table>
                    {% endif %}
                </details>
            </div>
            {% endfor %}
        </div>
        </body></html>
        """
    }),
    autoescape=select_autoescape(["html"]),
)

def render(name: str, **ctx):
    tmpl = templates.get_template(name)
    return HTMLResponse(tmpl.render(**ctx))

app = FastAPI(title="SAST SaaS MVP", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def index():
    # Render last 5 results
    items = store.all()[-5:][::-1]
    return render("index.html", items=items)

@app.post("/analyze")
async def analyze_endpoint(codefile: UploadFile = File(...), reqfile: UploadFile | None = File(None)):
    if not codefile.filename.endswith(".py"):
        raise HTTPException(400, "Only .py files supported")
    code = (await codefile.read()).decode("utf-8", errors="replace")
    result = analyze_code(code, filename=codefile.filename)

    # optional SBOM + vulns
    if reqfile is not None and reqfile.filename:
        reqtxt = (await reqfile.read()).decode("utf-8", errors="replace")
        sbom, vulns = await generate_sbom_and_vulns(reqtxt)
        result["sbom"] = sbom
        result["vulns"] = vulns

    store.add(result)
    return JSONResponse(result)

@app.post("/analyze_text")
async def analyze_text_endpoint(code: str = Form("")):
    result = analyze_code(code, filename="<pasted>")
    store.add(result)
    return JSONResponse(result)

@app.post("/analyze_dockerfile")
async def analyze_dockerfile_endpoint(dockerfile: UploadFile = File(...)):
    text = (await dockerfile.read()).decode("utf-8", errors="replace")
    findings = [
        {
            "rule_id": f.rule_id, "message": f.message, "severity": f.severity.value,
            "span": f.span.__dict__, "filename": f.filename
        } for f in scan_dockerfile(text, dockerfile.filename)
    ]
    res = {"filename": dockerfile.filename, "findings": findings, "taint_findings": [], "summary": "Dockerfile scan"}
    store.add(res)
    return JSONResponse(res)

@app.get("/results")
def results():
    return JSONResponse(store.all())

@app.get("/sarif")
def sarif():
    return JSONResponse(to_sarif(store.all()))

# -------------------------
# CLI
# -------------------------

def _print_json(obj):
    print(json.dumps(obj, indent=2))

def cli_scan(path: str):
    if not os.path.exists(path):
        print(f"No such file: {path}", file=sys.stderr); sys.exit(2)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()
    res = analyze_code(code, filename=os.path.basename(path))
    _print_json(res)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SAST SaaS MVP")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run API server")
    pscan = sub.add_parser("scan", help="scan a .py file")
    pscan.add_argument("path")
    args = ap.parse_args()
    if args.cmd == "scan":
        cli_scan(args.path)
    else:
        import uvicorn
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
