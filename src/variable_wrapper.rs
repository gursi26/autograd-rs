use crate::graph::Node;
use crate::ops::{BinaryOp, UnaryOp};
use crate::variable;
pub use crate::variable::Variable;

// Operable trait for wrapper functions
pub trait Operable<'a> {
    fn make_operable(self) -> Node<'a>;
}

// Implementations
impl<'a> Operable<'a> for Node<'a> {
    fn make_operable(self) -> Node<'a> {
        self
    }
}

impl<'a> Operable<'a> for &'a mut Variable {
    fn make_operable(self: &'a mut Variable) -> Node<'a> {
        Node::VariableNode {
            value: self.clone(),
            source: Some(self),
        }
    }
}

impl<'a> Operable<'a> for &'a Variable {
    fn make_operable(self: &'a Variable) -> Node<'a> {
        Node::VariableNode {
            value: self.clone(),
            source: None,
        }
    }
}

impl<'a> Operable<'a> for f64 {
    fn make_operable(self) -> Node<'a> {
        Node::VariableNode {
            value: variable::var!(self),
            source: None,
        }
    }
}

impl<'a> Operable<'a> for i32 {
    fn make_operable(self) -> Node<'a> {
        Node::VariableNode {
            value: variable::var!(self),
            source: None,
        }
    }
}

pub fn eval<'a>(mut root_node: Node<'a>) -> Variable {
    let (mut grad_ptrs, mut grad_values, output_value) = root_node.eval();
    for (ptr, grad) in grad_ptrs.drain(..).zip(grad_values.drain(..)) {
        ptr.grad = Some(grad);
    }
    output_value.clone()
}

pub fn add<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>,
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Add,
    }
}

pub fn mul<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>,
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Multiply,
    }
}

pub fn pow<'a, T, U>(lhs: T, rhs: U) -> Node<'a>
where
    T: Operable<'a>,
    U: Operable<'a>,
{
    let (lhs, rhs) = (lhs.make_operable(), rhs.make_operable());
    Node::BinaryNode {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        op: BinaryOp::Pow,
    }
}

pub fn negate<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryNode {
        value: Box::new(value),
        op: UnaryOp::Negate,
    }
}

pub fn exp<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryNode {
        value: Box::new(value),
        op: UnaryOp::Exp,
    }
}

pub fn reciprocal<'a, T: Operable<'a>>(value: T) -> Node<'a> {
    let value = value.make_operable();
    Node::UnaryNode {
        value: Box::new(value),
        op: UnaryOp::Reciprocal,
    }
}
