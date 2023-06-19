from __future__ import annotations
import numpy as np


class Tensor:
     
    def __init__(self, data: list[float], requires_grad: bool = False):
        self.data = np.array(data, dtype = float)
        self.requires_grad = requires_grad
        self.grad = None

    def __str__(self) -> str:
        return f"{{{self.data}, requires_grad = {self.requires_grad}}}"


class TensorNode:

    def __init__(self, data: Tensor):
        self.data = data

    def evaluate(self) -> tuple[Tensor, dict[Tensor, np.ndarray]]:
        if self.data.requires_grad:
            return (self.data, {self.data: np.ones_like(self.data.data)})
        return (self.data, {})


class UnaryOpNode:

    def __init__(self, data: TensorNode | UnaryOpNode | BinaryOpNode, op: str) -> None:
        self.data = data
        self.op = op

    def evaluate(self) -> tuple[Tensor, dict[Tensor, np.ndarray]]:
        evaluated, grad_dict = self.data.evaluate()
        if self.op == "-":
            evaluated.data *= -1
            for key in grad_dict.keys():
                grad_dict[key] *= -1.0

        elif self.op == "exp":
            for key in grad_dict.keys():
                grad_dict[key] *= np.exp(evaluated.data)
            evaluated.data = np.exp(evaluated.data)

        elif self.op == "reciprocal":
            for key in grad_dict.keys():
                grad_dict[key] *= (-1.0 / np.power(evaluated.data, 2))
            evaluated.data = 1 / evaluated.data

        elif self.op == "sum":
            evaluated.data = evaluated.data.sum()
            for key in grad_dict.keys():
                grad_dict[key] = grad_dict[key].sum()
        else:
            raise ValueError(f"Invalid operator {self.op}")
        return evaluated, grad_dict


class BinaryOpNode:

    def __init__(
            self, op: str, 
            rhs: TensorNode | UnaryOpNode | BinaryOpNode,
            lhs: TensorNode | UnaryOpNode | BinaryOpNode
            ) -> None:
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def evaluate(self) -> tuple[Tensor, dict[Tensor, np.ndarray]]:
        evaluated_rhs, rhs_grad_dict = self.rhs.evaluate()
        evaluated_lhs, lhs_grad_dict = self.lhs.evaluate()
        if self.op == "+":
            evaluated_lhs.data += evaluated_rhs.data

        elif self.op == "*":
            for key in rhs_grad_dict:
                rhs_grad_dict[key] *= evaluated_lhs.data
            for key in lhs_grad_dict:
                lhs_grad_dict[key] *= evaluated_rhs.data
            evaluated_lhs.data *= evaluated_rhs.data

        elif self.op == "pow":
            for key in rhs_grad_dict:
                rhs_grad_dict[key] *= (np.log(evaluated_lhs.data) * (evaluated_lhs.data ** evaluated_rhs.data))
            for key in lhs_grad_dict:
                lhs_grad_dict[key] *= (evaluated_rhs.data * (evaluated_lhs.data ** (evaluated_rhs.data - 1.0)))
            evaluated_lhs.data **= evaluated_rhs.data

        else:
            raise ValueError(f"Invalid operator {self.op}")

        out_dict = merge_grad_dict(rhs_grad_dict, lhs_grad_dict)
        return (evaluated_lhs, out_dict)


def merge_grad_dict(gd1: dict[Tensor, np.ndarray], gd2: dict[Tensor, np.ndarray]) -> dict[Tensor, np.ndarray]:
    (out_dict, to_iterate_dict) = (gd1, gd2) if len(gd1) > len(gd2) else (gd2, gd1)
    for key, value in to_iterate_dict.items():
        if key in out_dict:
            out_dict[key] += value
        else:
            out_dict[key] = value
    return out_dict


def convert(
    value : list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode
    ) -> TensorNode | UnaryOpNode | BinaryOpNode:
    if isinstance(value, list):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        value = Tensor(value)
    if isinstance(value, Tensor):
        value = TensorNode(value)
    return value


def sum(
    value: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> UnaryOpNode:
    return UnaryOpNode(
        op = "sum",
        data = convert(value),
    )

def add(
    rhs: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
    lhs: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> BinaryOpNode:
    return BinaryOpNode(
        op = "+",
        rhs = convert(rhs),
        lhs = convert(lhs)
    )


def mul(
    rhs: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
    lhs: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> BinaryOpNode:
    return BinaryOpNode(
        op = "*",
        rhs = convert(rhs),
        lhs = convert(lhs)
    )


def pow(
    lhs: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
    rhs: int | float | list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> BinaryOpNode:
    return BinaryOpNode(
        op = "pow",
        rhs = convert(rhs),
        lhs = convert(lhs)
    )


def neg(
    value: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> UnaryOpNode:
    return UnaryOpNode(
        op = "-",
        data = convert(value),
    )


def exp(
    value: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> UnaryOpNode:
    return UnaryOpNode(
        op = "exp",
        data = convert(value),
    )


def reciprocal(
    value: list[float | int] | np.ndarray | Tensor | TensorNode | UnaryOpNode | BinaryOpNode,
) -> UnaryOpNode:
    return UnaryOpNode(
        op = "reciprocal",
        data = convert(value),
    )


if __name__ == "__main__":
    # torch test
    import torch
    from torch import nn
    w = torch.tensor(1.0, requires_grad = True)
    b = torch.tensor(1.0, requires_grad = True)
    xvals = torch.tensor(list(range(1, 101)), dtype=float)
    yvals = torch.tensor(list(range(3, 202, 2)), dtype=float)
    crit = nn.MSELoss()
    loss = crit(yvals, w * xvals + b)
    loss.backward()
    print(loss, w.grad, b.grad)

    # custom autograd test
    w = Tensor([1.0 for _ in range(1, 101)], requires_grad = True)
    b = Tensor([1.0 for _ in range(1, 101)], requires_grad = True)
    xvals = Tensor(list(range(1, 101)))
    yvals = Tensor(list(range(3, 202, 2)))
    yhat = add(mul(xvals, w), b)
    mean_div_factor = Tensor([len(xvals.data)])
    pow_factor = Tensor([2 for _ in range(len(xvals.data))])
    loss, grad_dict = mul(reciprocal(mean_div_factor), sum(pow(add(yvals, neg(yhat)), pow_factor))).evaluate()
    print(loss.data, grad_dict[w], grad_dict[b])
