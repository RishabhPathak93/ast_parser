from typing import Dict, List, Any

def _loop_step(loop: Dict[str, Any]) -> str:
    if loop["type"] == "for":
        tgt = loop.get("target", "item")
        rng = loop.get("range")
        if rng:
            start, stop, step = rng
            # Describe the inclusive/exclusive nature 
            if step == 1:
                return f"Loop over {tgt} from {start} to {stop-1}."
            else:
                return f"Loop over {tgt} from {start} to {stop-1} stepping by {step}."
        else:
            return f"Loop over {tgt} in {loop.get('iter', 'iterable')}."
    if loop["type"] == "while":
        return f"Repeat while the condition ({loop.get('test','...')}) is true."
    return "A loop executes."

def _cond_step(cond: Dict[str, Any]) -> str:
    return f"If ({cond.get('test','condition')}) then execute the then-branch; otherwise the else-branch."

def _assign_steps(variables: Dict[str, List[str]]) -> List[str]:
    steps = []
    for name, vals in variables.items():
        if len(vals) == 1:
            steps.append(f"Set {name} = {vals[0]}.")
        else:
            steps.append(f"Update {name} sequentially: " + " → ".join(vals) + ".")
    return steps

def steps_from_facts(facts: Dict[str, Any]) -> List[str]:
    steps: List[str] = []
    # Vars
    steps.extend(_assign_steps(facts.get("variables", {})))
    # Loops
    for lp in facts.get("loops", []):
        steps.append(_loop_step(lp))
        if facts.get("prints"):
            # If there are prints and loop target exists, hint printed values
            tgt = lp.get("target")
            if tgt and any(isinstance(p, str) and tgt in p for p in facts["prints"]):
                rng = lp.get("range")
                if rng:
                    start, stop, step = rng
                    # show sample values (bounded)
                    if step == 0:
                        pass
                    else:
                        seq = list(range(start, stop, step))[:6]
                        steps.append(f"In each iteration, print {tgt} (e.g., {', '.join(map(str, seq))}).")
    # Conditionals
    for c in facts.get("conditionals", []):
        steps.append(_cond_step(c))
    # Prints
    for p in facts.get("prints", []):
        steps.append(f"Print {p}.")
    # Returns
    for r in facts.get("returns", []):
        steps.append(f"Return {r}.")
    return steps
