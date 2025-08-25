#!/usr/bin/env python3
import sys, json
from .ast_parser import analyze_code
from .step_extractor import steps_from_facts

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m code_explainer_ast.cli '<code or path.py>'", file=sys.stderr)
        sys.exit(1)
    arg = sys.argv[1]
    try:
        # If arg is a path to a file, read it
        import os
        if os.path.exists(arg) and os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8") as f:
                code = f.read()
        else:
            code = arg
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(2)

    facts = analyze_code(code)
    if "error" in facts:
        print(json.dumps(facts, indent=2))
        sys.exit(3)

    steps = steps_from_facts(facts)
    report = {
        "language": "python",
        "summary": facts.get("summary",""),
        "facts": facts,
        "steps": steps
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
