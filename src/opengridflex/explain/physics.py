def binding_constraints(result: dict, tol: float = 1e-5):
    """Placeholder: extract active/binding constraints from solver result metadata."""
    return [c for c in result.get("constraints", []) if abs(c.get("slack", 1.0)) <= tol]
