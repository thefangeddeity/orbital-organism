from __future__ import annotations

"""
Scratch-style block toolkit: the organism's path to proposing genuinely
NEW primitives, not just recombining a fixed human-written library.

modules/loss_blocks.py gave it 24 combinations to choose from -- all
hand-written. This gives it a small, fixed vocabulary of safe,
elementary operations (the "blocks", named after MIT Scratch's
drag-and-drop blocks: a child can't write a buffer overflow with them
because the vocabulary doesn't contain the concept, not because
anything is checking for one at runtime) that can be SNAPPED TOGETHER
into new expressions. A candidate is a tree of blocks -- JSON-
serializable, so a future Gemini-authored proposal is just data, never
executed Python source.

Safety is structural, not a filter:
    - The only node types the grammar can express are "op" (call a
      whitelisted function), "var" (read an input), and "const" (a
      number). There is no way to spell import, exec, eval, or a file
      path in this grammar -- not because they're blocked, but because
      the grammar has no node type that could hold them.
    - evaluate() and grad() never call Python's eval/exec on anything.
      They walk the tree and dispatch through a fixed lookup table.
    - Every block ships its own local derivative rule (this is where
      the actual "autodiff" is: reverse-mode backprop through a graph
      built entirely from blocks whose local Jacobian is known and
      small). check_gradients() finite-difference-verifies the WHOLE
      autodiff engine, not just each primitive in isolation, the way
      loss_blocks.py's check does -- catching a wiring bug in the tree
      walk itself, not just a wrong formula in one block.
    - Trees are bounded (MAX_NODES, MAX_DEPTH) before they're ever
      evaluated, so nothing this proposes can be arbitrarily expensive.

Usage sketch:
    tree = op("mul", op("abs", var("e")), const(2.0))
    validate(tree)                      # raises on anything malformed
    value = evaluate(tree, {"e": -3.0})           # 6.0
    grads = grad(tree, {"e": -3.0})               # {"e": -2.0}
"""

import math
from typing import Any

import numpy as np


# ======================================================================
# BLOCK VOCABULARY
#
# Each entry: (arity, forward_fn, local_grad_fn).
#
# forward_fn(*args) -> value, elementwise over numpy arrays.
# local_grad_fn(*args, value) -> tuple of d(value)/d(arg_i), one per
# argument, elementwise. value is passed in so ops that already
# computed something reusable (e.g. tanh) don't redo the work.
# ======================================================================

def _unary(fn, grad_fn):
    return (1, fn, grad_fn)


def _binary(fn, grad_fn):
    return (2, fn, grad_fn)


_EPS = 1e-12


BLOCKS: dict[str, tuple] = {
    # -- unary --
    "neg": _unary(
        lambda a: -a,
        lambda a, v: (-np.ones_like(a),),
    ),
    "abs": _unary(
        lambda a: np.abs(a),
        lambda a, v: (np.sign(a),),
    ),
    "square": _unary(
        lambda a: a * a,
        lambda a, v: (2.0 * a,),
    ),
    "sqrt": _unary(
        lambda a: np.sqrt(np.maximum(a, 0.0)),
        lambda a, v: (0.5 / np.maximum(v, _EPS),),
    ),
    "exp": _unary(
        # Clipped, not raw exp -- an unbounded proposed tree (e.g.
        # nested exp(exp(x))) must not be able to overflow to inf and
        # poison every downstream gradient with a NaN.
        lambda a: np.exp(np.clip(a, -30.0, 30.0)),
        lambda a, v: (v,),
    ),
    "log1p": _unary(
        lambda a: np.log1p(np.maximum(a, -1.0 + _EPS)),
        lambda a, v: (1.0 / np.maximum(1.0 + a, _EPS),),
    ),
    "sign": _unary(
        lambda a: np.sign(a),
        lambda a, v: (np.zeros_like(a),),  # 0 a.e., the honest answer
    ),
    "tanh": _unary(
        lambda a: np.tanh(a),
        lambda a, v: (1.0 - v * v,),
    ),
    "sigmoid": _unary(
        lambda a: 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0))),
        lambda a, v: (v * (1.0 - v),),
    ),
    # -- binary --
    "add": _binary(
        lambda a, b: a + b,
        lambda a, b, v: (np.ones_like(a), np.ones_like(b)),
    ),
    "sub": _binary(
        lambda a, b: a - b,
        lambda a, b, v: (np.ones_like(a), -np.ones_like(b)),
    ),
    "mul": _binary(
        lambda a, b: a * b,
        lambda a, b, v: (b, a),
    ),
    "div": _binary(
        lambda a, b: a / np.where(np.abs(b) < _EPS, _EPS, b),
        lambda a, b, v: (
            1.0 / np.where(np.abs(b) < _EPS, _EPS, b),
            -a / np.where(np.abs(b) < _EPS, _EPS, b) ** 2,
        ),
    ),
    "min": _binary(
        lambda a, b: np.minimum(a, b),
        lambda a, b, v: ((a <= b).astype(float), (a > b).astype(float)),
    ),
    "max": _binary(
        lambda a, b: np.maximum(a, b),
        lambda a, b, v: ((a >= b).astype(float), (a < b).astype(float)),
    ),
    "pow": _binary(
        # Bounded exponent -- an unbounded pow(x, huge) is the same
        # class of risk as unbounded exp.
        lambda a, b: np.sign(a) * np.abs(a) ** np.clip(b, 0.1, 6.0),
        lambda a, b, v: (
            np.clip(b, 0.1, 6.0) * np.abs(a) ** (np.clip(b, 0.1, 6.0) - 1.0),
            np.zeros_like(b),  # d/d(exponent) not needed -- b is a const in practice
        ),
    ),
}


MAX_NODES = 24
MAX_DEPTH = 6


# ======================================================================
# TREE CONSTRUCTION HELPERS
# ======================================================================

def op(name: str, *args) -> dict:
    return {"op": name, "args": list(args)}


def var(name: str) -> dict:
    return {"var": name}


def const(value: float) -> dict:
    return {"const": float(value)}


def to_expr_string(tree: dict) -> str:
    """
    Compact, human-readable rendering of a tree -- e.g.
    div(square(e), sqrt(add(1, square(e)))) -- for display (dashboard,
    logs), not for parsing anything back. Never used on the validation/
    evaluation path; a display bug here can't affect what's trusted.
    """
    if "op" in tree:
        args = ", ".join(to_expr_string(a) for a in tree["args"])
        return f"{tree['op']}({args})"
    if "var" in tree:
        return str(tree["var"])
    if "const" in tree:
        return f"{tree['const']:.3g}"
    return "?"


# ======================================================================
# VALIDATION
#
# The actual security boundary. Nothing downstream trusts a tree that
# hasn't passed this.
# ======================================================================

class InvalidBlockTree(ValueError):
    pass


def _walk_count(node: Any, depth: int = 0) -> int:
    if depth > MAX_DEPTH:
        raise InvalidBlockTree(f"tree exceeds MAX_DEPTH ({MAX_DEPTH})")

    if not isinstance(node, dict):
        raise InvalidBlockTree(f"node is not a dict: {node!r}")

    if "const" in node:
        if not isinstance(node["const"], (int, float)):
            raise InvalidBlockTree("const value must be a number")
        if len(node) != 1:
            raise InvalidBlockTree("const node must have no other keys")
        return 1

    if "var" in node:
        if not isinstance(node["var"], str):
            raise InvalidBlockTree("var name must be a string")
        if len(node) != 1:
            raise InvalidBlockTree("var node must have no other keys")
        return 1

    if "op" in node:
        name = node.get("op")
        if name not in BLOCKS:
            raise InvalidBlockTree(f"unknown op: {name!r}")

        args = node.get("args")
        if not isinstance(args, list):
            raise InvalidBlockTree("op node must have an 'args' list")

        arity, _, _ = BLOCKS[name]
        if len(args) != arity:
            raise InvalidBlockTree(
                f"op {name!r} expects {arity} args, got {len(args)}"
            )

        if set(node.keys()) - {"op", "args"}:
            raise InvalidBlockTree("op node has unexpected keys")

        count = 1
        for a in args:
            count += _walk_count(a, depth + 1)
        return count

    raise InvalidBlockTree(
        f"node has none of op/var/const: {list(node.keys())}"
    )


def validate(tree: Any) -> None:
    """Raises InvalidBlockTree if the tree is malformed, too deep, too
    large, or references anything outside the fixed BLOCKS vocabulary.
    Call this before evaluate()/grad() on anything not built locally
    by this module's own op()/var()/const() helpers -- in particular,
    always call it on anything deserialized from JSON."""
    node_count = _walk_count(tree)
    if node_count > MAX_NODES:
        raise InvalidBlockTree(
            f"tree has {node_count} nodes, exceeds MAX_NODES ({MAX_NODES})"
        )


# ======================================================================
# INTERPRETER: forward value + reverse-mode gradient
# ======================================================================

def evaluate(tree: dict, env: dict[str, Any]):
    """env maps var names to numpy arrays (or scalars)."""
    if "const" in tree:
        return np.asarray(tree["const"], dtype=float)

    if "var" in tree:
        name = tree["var"]
        if name not in env:
            raise KeyError(f"tree references undefined variable {name!r}")
        return np.asarray(env[name], dtype=float)

    name = tree["op"]
    _, forward, _ = BLOCKS[name]
    arg_values = [evaluate(a, env) for a in tree["args"]]
    return forward(*arg_values)


def _backward(tree: dict, env: dict, seed, grad_out: dict[str, Any]) -> None:
    """Accumulates d(root)/d(var) into grad_out, seeded by seed =
    d(root)/d(this node's value), given evaluate() has already been
    checked valid for this tree/env."""
    if "const" in tree:
        return

    if "var" in tree:
        name = tree["var"]
        grad_out[name] = grad_out.get(name, 0.0) + seed
        return

    name = tree["op"]
    _, forward, local_grad = BLOCKS[name]
    arg_values = [evaluate(a, env) for a in tree["args"]]
    value = forward(*arg_values)
    local = local_grad(*arg_values, value)

    for child, d_child in zip(tree["args"], local):
        _backward(child, env, seed * d_child, grad_out)


def grad(tree: dict, env: dict[str, Any]) -> dict[str, Any]:
    """Returns {var_name: d(tree)/d(var_name)}, one entry per distinct
    variable referenced in the tree. Re-evaluates the tree internally
    (this module favors a clear, obviously-correct implementation over
    a faster cache-forward-then-backward pass -- trees are capped at
    MAX_NODES=24, so the redundant forward evaluations are cheap)."""
    grad_out: dict[str, Any] = {}
    _backward(tree, env, np.ones_like(evaluate(tree, env)), grad_out)
    return grad_out


# ======================================================================
# ADAPTER: wrap a validated tree as a (value_fn, grad_fn) pair, the
# shape loss_blocks.ComposedLoss's custom_core expects -- this is the
# entire integration surface with the rest of the composer. Nothing on
# the loss_blocks.py side needs to know a tree exists.
# ======================================================================

def as_core_penalty(tree: dict):
    """
    Raises InvalidBlockTree via validate() if the tree is malformed --
    called once at proposal time, not on every training step, so this
    validation cost is paid once per candidate, not per batch.
    """
    validate(tree)

    def value_fn(e):
        return evaluate(tree, {"e": e})

    def grad_fn(e):
        return grad(tree, {"e": e}).get("e", np.zeros_like(e))

    return value_fn, grad_fn


# ======================================================================
# SELF-CHECK
#
# Verifies the interpreter's autodiff against numerical differentiation
# -- the same discipline loss_blocks.py applies to its hand-written
# primitives, applied here to the tree-walking engine itself. A bug in
# _backward()'s recursion would silently corrupt every future proposed
# block; this is what catches that before anything is trusted.
# ======================================================================

def check_gradients(tree: dict, env: dict[str, Any], tolerance: float = 1e-4) -> list[str]:
    validate(tree)

    analytic = grad(tree, env)
    failures = []
    step = 1e-6

    for name, value in env.items():
        if name not in analytic:
            continue  # tree doesn't use this variable; nothing to check

        value = np.asarray(value, dtype=float)
        numeric = np.zeros_like(value)

        for index in np.ndindex(value.shape) if value.shape else [()]:
            up = value.copy()
            down = value.copy()
            if index:
                up[index] += step
                down[index] -= step
            else:
                up = up + step
                down = down - step

            env_up = {**env, name: up}
            env_down = {**env, name: down}

            f_up = float(np.sum(evaluate(tree, env_up)))
            f_down = float(np.sum(evaluate(tree, env_down)))

            if index:
                numeric[index] = (f_up - f_down) / (2.0 * step)
            else:
                numeric = (f_up - f_down) / (2.0 * step)

        worst = float(np.max(np.abs(analytic[name] - numeric)))
        scale = max(1.0, float(np.max(np.abs(numeric))))

        if worst / scale > tolerance:
            failures.append(
                f"var {name!r}: max |analytic - numeric| = {worst:.3e} "
                f"(scaled {worst / scale:.3e})"
            )

    return failures


# ======================================================================
# PROOF OF EXPRESSIVENESS: recreate two of loss_blocks.py's hand-written
# primitives AS block trees, and cross-check they agree with the
# analytic versions already trusted there. If the toolkit can't
# reproduce what's already hand-verified, it can't be trusted to
# propose anything new.
# ======================================================================

def square_core_tree() -> dict:
    """square(e) -- should match loss_blocks._square exactly."""
    return op("square", var("e"))


def huber_like_tree(delta: float = 0.5) -> dict:
    """
    A smooth Huber-like block tree: delta^2 * (sqrt(1 + (e/delta)^2) - 1)
    -- this is actually loss_blocks._pseudo_huber, expressed as blocks
    instead of hand-written numpy, to prove the toolkit can reconstruct
    an existing, already-trusted primitive from elementary pieces.
    """
    e_over_delta = op("div", var("e"), const(delta))
    inside_sqrt = op("add", const(1.0), op("square", e_over_delta))
    return op(
        "mul",
        const(delta * delta),
        op("sub", op("sqrt", inside_sqrt), const(1.0)),
    )


def random_candidate_tree(rng: np.random.Generator, max_ops: int = 3) -> dict:
    """
    Generates a random small tree over the residual variable "e" --
    the organism's own bounded proposal mechanism, standing in for a
    future Gemini-authored one. Every tree this produces is, by
    construction, only ever built from op()/var()/const(), so it is
    automatically valid; validate() is still run on it before use as
    defense in depth, not because this generator is trusted to be
    correct forever.
    """
    unary_ops = [n for n, (arity, _, _) in BLOCKS.items() if arity == 1]
    binary_ops = [n for n, (arity, _, _) in BLOCKS.items() if arity == 2]

    def build(depth: int) -> dict:
        if depth <= 0 or rng.random() < 0.35:
            if rng.random() < 0.7:
                return var("e")
            return const(float(rng.uniform(0.1, 2.0)))

        if rng.random() < 0.5:
            name = unary_ops[rng.integers(len(unary_ops))]
            return op(name, build(depth - 1))

        name = binary_ops[rng.integers(len(binary_ops))]
        return op(name, build(depth - 1), build(depth - 1))

    tree = build(max_ops)
    validate(tree)
    return tree


if __name__ == "__main__":
    print("Checking hand-built example trees against numerical gradients...")

    e = np.array([-2.0, -0.3, 0.0, 0.4, 1.7])

    for label, tree in (
        ("square", square_core_tree()),
        ("pseudo-huber (via blocks)", huber_like_tree()),
    ):
        problems = check_gradients(tree, {"e": e})
        status = "OK" if not problems else "FAILED"
        print(f"  [{status}] {label}")
        for p in problems:
            print("     ", p)

    # Cross-check against loss_blocks.py's own hand-written primitives,
    # not just against finite differences -- two independently written
    # implementations of the same math agreeing is stronger evidence
    # than either one checked alone.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import loss_blocks

    block_square = evaluate(square_core_tree(), {"e": e})
    hand_square = loss_blocks._square(e)
    print(
        f"  square: block-tree vs hand-written match = "
        f"{np.allclose(block_square, hand_square)}"
    )

    block_huber = evaluate(huber_like_tree(), {"e": e})
    hand_huber = loss_blocks._pseudo_huber(e)
    print(
        f"  pseudo-huber: block-tree vs hand-written match = "
        f"{np.allclose(block_huber, hand_huber)}"
    )

    print()
    print("Random candidate generation + validation + gradient check "
          "(10 trees)...")
    rng = np.random.default_rng(7)
    ok_count = 0
    for i in range(10):
        tree = random_candidate_tree(rng)
        problems = check_gradients(tree, {"e": e})
        if not problems:
            ok_count += 1
        else:
            print(f"  tree {i}: {tree} -> {problems}")
    print(f"  {ok_count}/10 random trees passed the gradient check")
