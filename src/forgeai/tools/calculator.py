"""Arithmetic without eval: parse to an AST and walk only whitelisted nodes."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from forgeai.schemas.requirements import ToolHint
from forgeai.tools.base import ToolEnv, ToolResult, expect_args

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "round": lambda x, n=0: round(x, int(n)),
    "min": min,
    "max": max,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
}
MAX_LEN = 200
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15


class CalcError(ValueError):
    pass


def safe_eval(expression: str) -> float:
    expression = _THOUSANDS.sub("", expression)
    expression = expression.replace("\u00d7", "*").replace("\u00f7", "/").strip()
    if not expression or len(expression) > MAX_LEN:
        raise CalcError(f"expression must be 1-{MAX_LEN} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"not a valid arithmetic expression: {expression!r}") from exc
    return _eval(tree.body)


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):
            raise CalcError("booleans are not numbers")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError("exponent too large")
        if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
            raise CalcError("division by zero")
        result = _BINARY[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        result = _UNARY[type(node.op)](_eval(node.operand))
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        result = float(_FUNCS[node.func.id](*(_eval(a) for a in node.args)))
    else:
        raise CalcError(f"unsupported syntax: {type(node).__name__}")
    if isinstance(result, complex) or not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
        raise CalcError("result out of range")
    return result


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic only, e.g. '3 * 4200 + 0.18 * 12600'.")


class Calculator:
    name: ClassVar[str] = "calculator"
    capability: ClassVar[ToolHint] = "arithmetic"
    description: ClassVar[str] = (
        "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and "
        "round, min, max, abs, ceil, floor, sqrt. Use it for every calculation."
    )
    args_model: ClassVar[type[BaseModel]] = CalculatorArgs

    async def run(self, args: BaseModel, env: ToolEnv) -> ToolResult:
        args = expect_args(args, CalculatorArgs)
        try:
            value = safe_eval(args.expression)
        except CalcError as exc:
            return ToolResult.fail(str(exc))
        text = format_number(value)
        return ToolResult(ok=True, output=f"{args.expression} = {text}", data={"value": value})
