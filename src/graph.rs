use crate::variable::Variable;
use crate::ops::{UnaryOp, BinaryOp};
use crate::compute::*;

pub enum Node<'a> {
    VariableNode {
        value: Variable,
        source: Option<&'a mut Variable>
    },
    UnaryNode {
        op: UnaryOp,
        value: Box<Node<'a>>
    },
    BinaryNode {
        op: BinaryOp,
        lhs: Box<Node<'a>>,
        rhs: Box<Node<'a>>
    }
}

impl<'a> Node<'a> {
    pub fn eval(&mut self) -> (Vec<&mut Variable>, Vec<f64>, &mut Variable) {
        match self {
            Node::VariableNode { value, source } => {
                if let Some(s) = source {
                    if s.requires_grad {
                        return (vec![*s], vec![1.0], value);
                    }
                }
                return (Vec::new(), Vec::new(), value);
            },
            Node::UnaryNode { op, value } => {
                let (grad_ptrs, mut grad_values, value) = value.eval();
                match op {
                    UnaryOp::Exp => {
                        value.value = value.value.exp();
                        update_grads(&mut grad_values, value.value);
                    },
                    UnaryOp::Negate => {
                        update_grads(&mut grad_values, -1.0);
                        value.value *= -1.0;
                    },
                    UnaryOp::Reciprocal => {
                        update_grads(&mut grad_values, -1.0 / value.value.powi(2));
                        value.value = 1.0 / value.value;
                    }
                };
                return (grad_ptrs, grad_values, value);
            },
            Node ::BinaryNode { op, lhs, rhs } => {
                let (rhs_ptrs, mut rhs_grads, rhs_value) = rhs.eval();
                let (mut lhs_ptrs, mut lhs_grads, lhs_value) = lhs.eval();

                match op {
                    BinaryOp::Add => {
                        lhs_value.value += rhs_value.value;
                    },
                    BinaryOp::Pow => {
                        update_grads(&mut rhs_grads, lhs_value.value.ln() * lhs_value.value.powf(rhs_value.value));
                        update_grads(&mut lhs_grads, rhs_value.value * lhs_value.value.powf(rhs_value.value - 1.0));
                        lhs_value.value = lhs_value.value.powf(rhs_value.value);
                    },
                    BinaryOp::Multiply => {
                        update_grads(&mut rhs_grads, lhs_value.value);
                        update_grads(&mut lhs_grads, rhs_value.value);
                        lhs_value.value *= rhs_value.value;
                    }
                }
                compute_merged_grads(rhs_ptrs, &mut lhs_ptrs, rhs_grads, &mut lhs_grads);
                return (lhs_ptrs, lhs_grads, lhs_value);
            }
        }
    }
}
