import ast
from typing import Any, Dict, List, Optional, Tuple

# Utility: safe unparse for Python 3.9+ (ast.unparse exists in 3.9+)
try:
    from ast import unparse as ast_unparse
except ImportError:  # pragma: no cover
    # Fallback minimal unparse
    def ast_unparse(node):
        return "<expr>"
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
    return node.__class__.__name__

def _is_print_call(node: ast.Call) -> bool:
    return _name_of(node.func) == "print"

def _parse_range_args(call: ast.Call) -> Optional[Tuple[int,int,int]]:
    if _name_of(call.func) != "range":
        return None
    # Only handle literal ints for static inference
    vals = []
    for arg in call.args:
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub) and isinstance(arg.operand, ast.Constant) and isinstance(arg.operand.value, int):
            vals.append(-arg.operand.value)
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            vals.append(arg.value)
        else:
            return None
    if len(vals) == 1:
        return (0, vals[0], 1)
    if len(vals) == 2:
        return (vals[0], vals[1], 1)
    if len(vals) >= 3:
        return (vals[0], vals[1], vals[2])
    return None

class CodeFactsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.facts: Dict[str, Any] = {
            "imports": [],
            "variables": {},            # name -> list of assignments (strings)
            "loops": [],                # {type, target, iter, range?, body_len}
            "conditionals": [],         # {type, test, body_len, orelse_len}
            "functions": [],            # {name, args, returns, docstring}
            "calls": [],                # called function names
            "prints": [],               # printed expressions (strings)
            "returns": [],              # return expressions (strings)
            "exceptions": [],           # try/except structure
            "comprehensions": [],       # list/dict/set comprehensions summary
            "literals": [],             # noteworthy literals
            "lines": 0,
        }
        self._in_function: Optional[str] = None

    def visit_Module(self, node: ast.Module):
        self.facts["lines"] = getattr(node, "end_lineno", None) or 0
        self.generic_visit(node)

    # Imports
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.facts["imports"].append({"module": alias.name, "as": alias.asname})
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            self.facts["imports"].append({"from": node.module, "name": alias.name, "as": alias.asname})
        self.generic_visit(node)

    # Assignments
    def visit_Assign(self, node: ast.Assign):
        val = ast_unparse(node.value)
        for tgt in node.targets:
            name = _name_of(tgt)
            self.facts["variables"].setdefault(name, []).append(val)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        name = _name_of(node.target)
        val = ast_unparse(node.value) if node.value is not None else "None"
        ann = ast_unparse(node.annotation) if node.annotation is not None else ""
        self.facts["variables"].setdefault(name, []).append(f"{val} : {ann}")
        self.generic_visit(node)

    # Loops
    def visit_For(self, node: ast.For):
        target = _name_of(node.target)
        iter_str = ast_unparse(node.iter)
        rng = None
        if isinstance(node.iter, ast.Call):
            rng = _parse_range_args(node.iter)
        self.facts["loops"].append({
            "type": "for",
            "target": target,
            "iter": iter_str,
            "range": rng,
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0
        })
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        test = ast_unparse(node.test)
        self.facts["loops"].append({
            "type": "while",
            "test": test,
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0
        })
        self.generic_visit(node)

    # Conditionals
    def visit_If(self, node: ast.If):
        test = ast_unparse(node.test)
        self.facts["conditionals"].append({
            "type": "if",
            "test": test,
            "body_len": len(node.body),
            "orelse_len": len(node.orelse) if node.orelse else 0
        })
        self.generic_visit(node)

    # Functions
    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = [a.arg for a in node.args.args]
        docstring = ast.get_docstring(node)
        returns = ast_unparse(node.returns) if node.returns else None
        self.facts["functions"].append({
            "name": node.name, "args": args, "returns": returns, "docstring": docstring
        })
        prev = self._in_function; self._in_function = node.name
        self.generic_visit(node)
        self._in_function = prev

    def visit_Return(self, node: ast.Return):
        self.facts["returns"].append(ast_unparse(node.value) if node.value else "None")
        self.generic_visit(node)

    # Calls and print
    def visit_Call(self, node: ast.Call):
        fname = _name_of(node.func)
        self.facts["calls"].append(fname)
        if _is_print_call(node):
            args = [ast_unparse(a) for a in node.args]
            self.facts["prints"].append(args if len(args) > 1 else (args[0] if args else ""))
        self.generic_visit(node)

    # Exceptions
    def visit_Try(self, node: ast.Try):
        entry = {
            "handlers": [{"type": _name_of(h.type) if h.type else "Exception", "name": h.name} for h in node.handlers],
            "finalbody": bool(node.finalbody),
            "orelse": bool(node.orelse)
        }
        self.facts["exceptions"].append(entry)
        self.generic_visit(node)

    # Comprehensions
    def visit_ListComp(self, node: ast.ListComp):
        self.facts["comprehensions"].append({
            "type": "list", "elt": ast_unparse(node.elt), "generators": len(node.generators)
        })
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        self.facts["comprehensions"].append({
            "type": "dict", "key": ast_unparse(node.key), "value": ast_unparse(node.value),
            "generators": len(node.generators)
        })
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.facts["comprehensions"].append({
            "type": "set", "elt": ast_unparse(node.elt), "generators": len(node.generators)
        })
        self.generic_visit(node)

    # Literals (collect notable constants)
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float, str)) and len(self.facts["literals"]) < 128:
            self.facts["literals"].append(node.value)
        # no generic_visit needed for constants

def analyze_code(code: str) -> Dict[str, Any]:
    """
    Analyze Python code and return structured facts + a neutral summary.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e}"}

    visitor = CodeFactsVisitor()
    visitor.visit(tree)
    facts = visitor.facts

    # Quick neutral summary
    parts = []
    if facts["functions"]:
        fnames = ", ".join(fn["name"] for fn in facts["functions"])
        parts.append(f"Defines functions: {fnames}.")
    if facts["loops"]:
        loop_kinds = ", ".join(l["type"] for l in facts["loops"])
        parts.append(f"Contains loops ({loop_kinds}).")
    if facts["conditionals"]:
        parts.append("Contains conditional logic.")
    if facts["prints"]:
        parts.append("Produces console output.")
    if not parts:
        parts.append("Simple sequence of statements.")

    facts["summary"] = " ".join(parts)
    return facts
