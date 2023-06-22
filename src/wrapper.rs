use crate::tensor::Tensor;
use crate::graph::Node;
use crate::ops::{BinaryOp, UnaryOp};

pub trait Operable<'a> {
    fn make_operable(self) -> Node<'a>;
}

impl<'a> Operable<'a> for Node<'a> {
    fn make_operable(self) -> Node<'a> {
        self
    }
}

impl <'a> Operable<'a> for &'a mut Tensor {
    fn make_operable(self: &'a mut Tensor) -> Node<'a> {
        Node::TensorNode { tensor: self.clone(), source: self }
    }
}

pub fn eval<'a>(mut root_node: Node<'a>) -> Tensor {
    let (mut grad_ptrs, mut grad_values, output_tensor) = root_node.eval();
    for (ptr, grad) in grad_ptrs.drain(..).zip(grad_values.drain(..)) {
        ptr.grad = Some(grad);
    }
    output_tensor.clone()
}

pub fn add<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryOpNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Add
    }
}

pub fn mul<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryOpNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Multiply
    }
}

pub fn pow<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryOpNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Pow
    }
}

pub fn negate<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryOpNode {
        value: Box::new(value),
        op: UnaryOp::Negate
    }
}

pub fn exp<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryOpNode {
        value: Box::new(value),
        op: UnaryOp::Exp
    }
}

pub fn reciprocal<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryOpNode {
        value: Box::new(value),
        op: UnaryOp::Reciprocal
    }
}

pub fn sum<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryOpNode {
        value: Box::new(value),
        op: UnaryOp::Sum
    }
}
