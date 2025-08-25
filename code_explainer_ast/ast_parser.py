import ast
import builtins
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _safe_unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        from ast import unparse as ast_unparse  # type: ignore
        return ast_unparse(node)
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


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class Span:
    line: int
    col: int
    end_line: Optional[int] = None
    end_col: Optional[int] = None


@dataclass
class Finding:
    rule_id: str
    message: str
    severity: str  # "info" | "warning" | "error"
    span: Span
    function: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleContext:
    tree: ast.AST
    filename: str
    call_graph: Dict[str, Set[str]]
    symbol_table: Dict[str, Set[int]]
    assigned_in_function: Dict[str, Set[str]]
    imported: Dict[str, Set[str]]


class Rule:
    id: str = "GENERIC"
    description: str = ""
    severity: str = "warning"

    def check(self, ctx: RuleContext) -> List[Finding]:  # override
        return []


# ------------------------------------------------------------
# Core Visitor to collect rich facts
# ------------------------------------------------------------

class FactsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.facts: Dict[str, Any] = {
            "imports": [],
            "variables": {},  # name -> list of assignments (strings)
            "loops": [],
            "conditionals": [],
            "functions": [],
            "calls": [],
            "prints": [],
            "returns": [],
            "exceptions": [],
            "comprehensions": [],
            "literals": [],
            "lines": 0,
            "call_graph": {},  # fn -> set(callees)
        }
        self.current_function: Optional[str] = None
        self._function_stack: List[str] = []
        self.assigned_in_function: Dict[str, Set[str]] = {}
        self.symbol_table: Dict[str, Set[int]] = {}  # name -> set(line nos where referenced)
        self.imported: Dict[str, Set[str]] = {}      # module/name -> aliases

    # Helper
    def _span(self, node: ast.AST) -> Span:
        return Span(
            line=getattr(node, "lineno", 0) or 0,
            col=getattr(node, "col_offset", 0) or 0,
            end_line=getattr(node, "end_lineno", None),
            end_col=getattr(node, "end_col_offset", None),
        )

    def visit_Module(self, node: ast.Module):
        self.facts["lines"] = getattr(node, "end_lineno", None) or 0
        self.generic_visit(node)

    # Imports
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.facts["imports"].append({"module": alias.name, "as": alias.asname, "span": self._span(node).__dict__})
            self.imported.setdefault(alias.name, set()).add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self.facts["imports"].append({"from": node.module, "name": alias.name, "as": alias.asname, "span": self._span(node).__dict__})
            key = f"{node.module}.{alias.name}" if node.module else alias.name
            self.imported.setdefault(key, set()).add(alias.asname or alias.name)
        self.generic_visit(node)

    # Assignments
    def visit_Assign(self, node: ast.Assign):
        val = _safe_unparse(node.value)
        for tgt in node.targets:
            name = _name_of(tgt)
            self.facts["variables"].setdefault(name, []).append({"value": val, "span": self._span(node).__dict__})
            if self.current_function:
                self.assigned_in_function.setdefault(self.current_function, set()).add(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        name = _name_of(node.target)
        val = _safe_unparse(node.value) if node.value is not None else "None"
        ann = _safe_unparse(node.annotation) if node.annotation is not None else ""
        self.facts["variables"].setdefault(name, []).append({"value": f"{val} : {ann}", "span": self._span(node).__dict__})
        if self.current_function:
            self.assigned_in_function.setdefault(self.current_function, set()).add(name)
        self.generic_visit(node)

    # Loops
    def visit_For(self, node: ast.For):
        self.facts["loops"].append({
            "type": "for",
            "target": _name_of(node.target),
            "iter": _safe_unparse(node.iter),
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0,
            "span": self._span(node).__dict__
        })
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.facts["loops"].append({
            "type": "while",
            "test": _safe_unparse(node.test),
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0,
            "span": self._span(node).__dict__
        })
        self.generic_visit(node)

    # Conditionals
    def visit_If(self, node: ast.If):
        self.facts["conditionals"].append({
            "type": "if",
            "test": _safe_unparse(node.test),
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0,
            "span": self._span(node).__dict__
        })
        self.generic_visit(node)

    # Functions
    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = [a.arg for a in node.args.args]
        returns = _safe_unparse(node.returns) if node.returns else None
        docstring = ast.get_docstring(node)
        name = node.name
        self.facts["functions"].append({
            "name": name,
            "args": args,
            "returns": returns,
            "docstring": docstring,
            "span": self._span(node).__dict__,
            "cyclomatic_complexity": self._cyclomatic_complexity(node),
        })
        self._function_stack.append(name)
        prev = self.current_function
        self.current_function = name
        self.assigned_in_function.setdefault(name, set())
        self.facts["call_graph"].setdefault(name, set())
        self.generic_visit(node)
        self.current_function = prev
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Treat similar to FunctionDef
        return self.visit_FunctionDef(node)  # type: ignore

    def _cyclomatic_complexity(self, fn: ast.FunctionDef) -> int:
        # Very simple CC = 1 + number of decision points
        decision_nodes = (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler, ast.Try, ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.Match)
        count = 1
        for n in ast.walk(fn):
            if isinstance(n, decision_nodes):
                count += 1
        return count

    def visit_Return(self, node: ast.Return):
        self.facts["returns"].append({"expr": _safe_unparse(node.value) if node.value else "None", "span": self._span(node).__dict__, "function": self.current_function})
        self.generic_visit(node)

    # Calls and print
    def visit_Call(self, node: ast.Call):
        fname = _name_of(node.func)
        self.facts["calls"].append({"name": fname, "args": [_safe_unparse(a) for a in node.args], "span": self._span(node).__dict__, "function": self.current_function})
        if self.current_function:
            self.facts["call_graph"].setdefault(self.current_function, set()).add(fname)
        # capture print outputs
        if fname == "print":
            args = [_safe_unparse(a) for a in node.args]
            self.facts["prints"].append({"values": args, "span": self._span(node).__dict__})
        self.generic_visit(node)

    # Exceptions
    def visit_Try(self, node: ast.Try):
        entry = {
            "handlers": [{"type": _name_of(h.type) if h.type else "Exception", "name": h.name} for h in node.handlers],
            "finalbody": bool(node.finalbody),
            "orelse": bool(node.orelse),
            "span": self._span(node).__dict__,
        }
        self.facts["exceptions"].append(entry)
        self.generic_visit(node)

    # Comprehensions
    def visit_ListComp(self, node: ast.ListComp):
        self.facts["comprehensions"].append({"type": "list", "elt": _safe_unparse(node.elt), "generators": len(node.generators), "span": self._span(node).__dict__})
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        self.facts["comprehensions"].append({"type": "dict", "key": _safe_unparse(node.key), "value": _safe_unparse(node.value), "generators": len(node.generators), "span": self._span(node).__dict__})
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.facts["comprehensions"].append({"type": "set", "elt": _safe_unparse(node.elt), "generators": len(node.generators), "span": self._span(node).__dict__})
        self.generic_visit(node)

    # Literals & symbols
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float, str)) and len(self.facts["literals"]) < 256:
            self.facts["literals"].append({"value": node.value, "span": self._span(node).__dict__})

    def visit_Name(self, node: ast.Name):
        self.symbol_table.setdefault(node.id, set()).add(getattr(node, "lineno", 0) or 0)
        self.generic_visit(node)


# ------------------------------------------------------------
# Simple Taint Analysis (flow-insensitive, intraprocedural)
# ------------------------------------------------------------

TAINT_SOURCES = {
    "input",
}

# potential unsafe sinks
TAINT_SINKS = {
    "eval", "exec", "os.system", "subprocess.call", "subprocess.Popen",
    "cursor.execute", "cursor.executemany",
}


def _maybe_qualname(call: ast.Call) -> str:
    return _name_of(call.func)


class TaintVisitor(ast.NodeVisitor):
    def __init__(self):
        self.tainted: Set[str] = set()
        self.findings: List[Finding] = []
        self.current_function: Optional[str] = None

    def _span(self, node: ast.AST) -> Span:
        return Span(
            line=getattr(node, "lineno", 0) or 0,
            col=getattr(node, "col_offset", 0) or 0,
            end_line=getattr(node, "end_lineno", None),
            end_col=getattr(node, "end_col_offset", None),
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign):
        # x = input() -> x tainted
        is_tainted = self._expr_is_tainted(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and is_tainted:
                self.tainted.add(tgt.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = _maybe_qualname(node)
        # Mark taint from direct sources
        if name in TAINT_SOURCES:
            # propagate to assignment handled in Assign
            pass
        # Sinks: if any arg tainted, report
        if name in TAINT_SINKS:
            if any(self._expr_is_tainted(arg) for arg in node.args):
                self.findings.append(Finding(
                    rule_id="TAINT-001",
                    message=f"Tainted data flows into sink '{name}'.",
                    severity="error",
                    span=self._span(node),
                    function=self.current_function,
                ))
        self.generic_visit(node)

    def _expr_is_tainted(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and _maybe_qualname(node) in TAINT_SOURCES:
            return True
        if isinstance(node, ast.Name):
            return node.id in self.tainted
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.JoinedStr, ast.FormattedValue)):
            return any(self._expr_is_tainted(c) for c in ast.iter_child_nodes(node))
        return False


# ------------------------------------------------------------
# Built-in Rule Implementations
# ------------------------------------------------------------

class RuleMutableDefaultArgs(Rule):
    id = "PY1001"
    description = "Avoid mutable default arguments (list/dict/set)."
    severity = "warning"

    def check(self, ctx: RuleContext) -> List[Finding]:
        findings: List[Finding] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(Finding(
                            rule_id=self.id,
                            message=f"Function '{node.name}' has mutable default argument.",
                            severity=self.severity,
                            span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                            function=node.name,
                        ))
        return findings


class RuleBareExcept(Rule):
    id = "PY1002"
    description = "Avoid bare except; catch specific exceptions."
    severity = "warning"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                out.append(Finding(
                    rule_id=self.id,
                    message="Bare except detected; specify an exception type.",
                    severity=self.severity,
                    span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                ))
        return out


class RuleBroadException(Rule):
    id = "PY1003"
    description = "Avoid catching 'Exception' or 'BaseException' unless necessary."
    severity = "info"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                out.append(Finding(
                    rule_id=self.id,
                    message=f"Catching broad exception '{node.type.id}'.",
                    severity=self.severity,
                    span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                ))
        return out


class RuleRiskyCalls(Rule):
    id = "PY1004"
    description = "Flag risky calls like eval/exec/os.system/subprocess.*"
    severity = "warning"

    RISKY = {"eval", "exec", "os.system", "subprocess.call", "subprocess.Popen"}

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                name = _name_of(node.func)
                if name in self.RISKY:
                    out.append(Finding(
                        rule_id=self.id,
                        message=f"Risky call '{name}'.",
                        severity=self.severity,
                        span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                    ))
        return out


class RulePrintInLibrary(Rule):
    id = "PY1005"
    description = "Avoid print statements in library code (prefer logging)."
    severity = "info"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call) and _name_of(node.func) == "print":
                out.append(Finding(
                    rule_id=self.id,
                    message="print() found; consider using logging.",
                    severity=self.severity,
                    span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                ))
        return out


class RuleUnreachableAfterReturn(Rule):
    id = "PY1006"
    description = "Detect unreachable code after return/raise/break/continue."
    severity = "warning"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        terminators = (ast.Return, ast.Raise, ast.Break, ast.Continue)

        def scan_block(stmts: List[ast.stmt]):
            unreachable = False
            for i, s in enumerate(stmts):
                if unreachable:
                    out.append(Finding(
                        rule_id=self.id,
                        message="Unreachable statement.",
                        severity=self.severity,
                        span=Span(s.lineno, s.col_offset, getattr(s, 'end_lineno', None), getattr(s, 'end_col_offset', None)),
                    ))
                if isinstance(s, terminators):
                    unreachable = True
                # Recurse into nested blocks
                for child in ast.iter_child_nodes(s):
                    if isinstance(child, ast.stmt):
                        # Handle common body attributes
                        for field in ("body", "orelse", "finalbody"):
                            body = getattr(s, field, None)
                            if isinstance(body, list):
                                scan_block(body)
        # top-level blocks
        for node in ast.walk(ctx.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan_block(node.body)
        return out


class RuleShadowBuiltins(Rule):
    id = "PY1007"
    description = "Variable or function shadows a Python builtin."
    severity = "info"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        builtin_names = set(dir(builtins))
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in builtin_names:
                    out.append(Finding(
                        rule_id=self.id,
                        message=f"Function name '{node.name}' shadows builtin.",
                        severity=self.severity,
                        span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                        function=node.name,
                    ))
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in builtin_names:
                        out.append(Finding(
                            rule_id=self.id,
                            message=f"Variable '{tgt.id}' shadows builtin.",
                            severity=self.severity,
                            span=Span(node.lineno, node.col_offset, getattr(node, 'end_lineno', None), getattr(node, 'end_col_offset', None)),
                        ))
        return out


class RuleUnusedImports(Rule):
    id = "PY1008"
    description = "Detect unused imports (approximate)."
    severity = "info"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        # If an imported alias never appears in symbol_table beyond its import line, flag it.
        referenced = ctx.symbol_table
        for imp_key, aliases in ctx.imported.items():
            for alias in aliases:
                lines = referenced.get(alias, set())
                if not lines:
                    # Find an import node span to report (best-effort)
                    span = Span(1, 0)
                    out.append(Finding(
                        rule_id=self.id,
                        message=f"Imported '{alias}' appears unused.",
                        severity=self.severity,
                        span=span,
                    ))
        return out


class RuleUnusedVariables(Rule):
    id = "PY1009"
    description = "Detect variables assigned but not used (within a function, approximate)."
    severity = "info"

    def check(self, ctx: RuleContext) -> List[Finding]:
        out: List[Finding] = []
        for fn, assigned in ctx.assigned_in_function.items():
            for var in assigned:
                refs = ctx.symbol_table.get(var, set())
                # If a name is referenced only on its assignment line, treat as unused (approximate)
                if len(refs) <= 1:
                    out.append(Finding(
                        rule_id=self.id,
                        message=f"Variable '{var}' may be unused in function '{fn}'.",
                        severity=self.severity,
                        span=Span(1, 0),
                        function=fn,
                    ))
        return out


# Registry of rules
BUILTIN_RULES: List[Rule] = [
    RuleMutableDefaultArgs(),
    RuleBareExcept(),
    RuleBroadException(),
    RuleRiskyCalls(),
    RulePrintInLibrary(),
    RuleUnreachableAfterReturn(),
    RuleShadowBuiltins(),
    RuleUnusedImports(),
    RuleUnusedVariables(),
]


# ------------------------------------------------------------
# Analyzer API
# ------------------------------------------------------------

def analyze_code_advanced(code: str, filename: str = "<memory>") -> Dict[str, Any]:
    """
    Parse Python code, collect rich facts, run lint rules & simple taint analysis.
    Returns a dict with keys: facts, findings, taint_findings, summary.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}", "filename": filename}

    facts_visitor = FactsVisitor()
    facts_visitor.visit(tree)

    facts = facts_visitor.facts
    # Convert call_graph sets to lists for JSON-compat
    facts["call_graph"] = {k: sorted(list(v)) for k, v in facts.get("call_graph", {}).items()}

    # Build RuleContext
    ctx = RuleContext(
        tree=tree,
        filename=filename,
        call_graph={k: set(v) for k, v in facts["call_graph"].items()},
        symbol_table=facts_visitor.symbol_table,
        assigned_in_function=facts_visitor.assigned_in_function,
        imported=facts_visitor.imported,
    )

    # Run rules
    findings: List[Finding] = []
    for rule in BUILTIN_RULES:
        findings.extend(rule.check(ctx))

    # Simple taint analysis
    taint = TaintVisitor()
    taint.visit(tree)

    # Neutral summary
    parts: List[str] = []
    if facts["functions"]:
        fnames = ", ".join(f["name"] for f in facts["functions"])
        parts.append(f"Defines functions: {fnames}.")
    if facts["loops"]:
        kinds = ", ".join(l["type"] for l in facts["loops"]) or ""
        parts.append(f"Contains loops ({kinds}).")
    if facts["conditionals"]:
        parts.append("Contains conditional logic.")
    if facts["prints"]:
        parts.append("Produces console output.")
    if not parts:
        parts.append("Simple sequence of statements.")

    # Serialize findings
    def _ser_f(f: Finding) -> Dict[str, Any]:
        return {
            "rule_id": f.rule_id,
            "message": f.message,
            "severity": f.severity,
            "span": f.span.__dict__,
            "function": f.function,
            "extra": f.extra,
        }

    result = {
        "filename": filename,
        "facts": facts,
        "findings": [_ser_f(f) for f in findings],
        "taint_findings": [_ser_f(f) for f in taint.findings],
        "summary": " ".join(parts),
        "metrics": {
            "functions": len(facts.get("functions", [])),
            "loops": len(facts.get("loops", [])),
            "conditionals": len(facts.get("conditionals", [])),
            "cyclomatic_complexity_avg": _avg([f.get("cyclomatic_complexity", 1) for f in facts.get("functions", [])])
        },
    }
    return result


def _avg(nums: List[int]) -> float:
    return round(sum(nums) / len(nums), 2) if nums else 0.0


# ------------------------------------------------------------
# CLI helper (optional)
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys
    p = argparse.ArgumentParser(description="Advanced static analyzer (AST)")
    p.add_argument("path", nargs="?", help="Path to .py file (reads stdin if omitted)")
    args = p.parse_args()

    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            code = f.read()
        res = analyze_code_advanced(code, filename=args.path)
    else:
        code = sys.stdin.read()
        res = analyze_code_advanced(code)

    print(json.dumps(res, indent=2))
