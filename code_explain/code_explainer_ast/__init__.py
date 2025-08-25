"""
code_explainer_ast: Static AST analyzer for step-by-step code explanations.

Usage:
    from code_explainer_ast.ast_parser import analyze_code
    report = analyze_code("for i in range(3): print(i)")
    print(report["summary"])
""" 
from .ast_parser import analyze_code
from .step_extractor import steps_from_facts
