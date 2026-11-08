"""Versioned, bounded arithmetic policies; model output is never Python code."""
import ast
import math
import operator

COMPONENTS = ('task', 'pair', 'charge', 'imbalance', 'move', 'energy_risk')
VARIABLES = (*COMPONENTS, 'workers', 'uavs', 'cars', 'remaining', 'time_left',
             'triples', 'mean_duration', 'mean_energy', 'mean_slack')
FUNCTIONS = {'min': min, 'max': max, 'abs': abs, 'sqrt': math.sqrt,
             'log1p': math.log1p, 'tanh': math.tanh}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


def expression(source, values):
    if not isinstance(source, str) or len(source) > 400:
        raise ValueError('Formula must be a string of at most 400 characters')
    tree = ast.parse(source, mode='eval')
    if len(list(ast.walk(tree))) > 100:
        raise ValueError('Formula too complex')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.Name) and node.id in VARIABLES:
            result = float(values[node.id])
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            result = OPS[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FUNCTIONS and not node.keywords and 1 <= len(node.args) <= 4):
            result = FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        else:
            raise ValueError('Unsupported formula syntax')
        if not math.isfinite(result) or abs(result) > 1e9:
            raise ValueError('Nonfinite or excessive formula value')
        return result
    return visit(tree.body)


def default_policy():
    return {'formulas': {name: name for name in COMPONENTS},
            'weights': dict(zip(COMPONENTS, (.25, .50, .15, .04, .03, .03)))}


def validate(policy):
    if not isinstance(policy, dict) or set(policy) != {'formulas', 'weights'}:
        raise ValueError('Expected formulas and weights')
    for section in policy.values():
        if not isinstance(section, dict) or set(section) != set(COMPONENTS):
            raise ValueError('All six components are required')
    for weight in policy['weights'].values():
        if type(weight) not in (float, int) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Weights must be finite numbers in [0,1]')
    if abs(sum(policy['weights'].values()) - 1) > 1e-6:
        raise ValueError('Weights must sum to 1')
    for formula in policy['formulas'].values():
        for value in (0., 1., 100.):
            if expression(formula, dict.fromkeys(VARIABLES, value)) < 0:
                raise ValueError('Components must be nonnegative')
    return policy


def rank(candidates, policy):
    transformed = []
    for candidate in candidates:
        parts = {k: expression(policy['formulas'][k], candidate['features']) for k in COMPONENTS}
        if any(v < 0 for v in parts.values()):
            raise ValueError('Negative component on actual state')
        transformed.append(parts)
    maxima = {k: max((c[k] for c in transformed), default=0) for k in COMPONENTS[:3]}
    result = []
    for candidate, parts in zip(candidates, transformed):
        normalized = {k: parts[k] / (maxima[k] + 1e-8) if k in maxima
                      else min(1., parts[k]) for k in COMPONENTS}
        score = sum(policy['weights'][k] * normalized[k] * (1 if i < 3 else -1)
                    for i, k in enumerate(COMPONENTS))
        result.append({**candidate, 'components': parts, 'normalized': normalized, 'score': score})
    return sorted(result, key=lambda c: (-c['score'], -c['features']['pair'],
                  -c['features']['task'], -c['features']['charge'],
                  c['features']['move'], *c['position']))
