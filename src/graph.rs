use crate::tensor::Tensor;
use crate::ops::{UnaryOp, BinaryOp};
use crate::compute::*;

pub enum Node<'a> {
    TensorNode {
        tensor: Tensor,
        source: &'a mut Tensor,
    },
    UnaryOpNode {
        value: Box<Node<'a>>,
        op: UnaryOp
    },
    BinaryOpNode {
        rhs: Box<Node<'a>>,
        lhs: Box<Node<'a>>,
        op: BinaryOp
    }
}


impl<'a> Node<'a> {
    pub fn eval(&mut self) -> (Vec<&mut Tensor>, Vec<Vec<f64>>, &mut Tensor) {
        match self {
            Node::TensorNode { tensor, source } => {
                if source.requires_grad {
                    return (vec![*source], vec![vec![1.0; tensor.length]], tensor);
                } else {
                    return (Vec::new(), Vec::new(), tensor);
                }
            },
            Node::UnaryOpNode { value, op } => {
                let (grad_ptrs, mut grad_values, tensor) = value.eval();
                match op {
                    UnaryOp::Negate => {
                        for v in grad_values.iter_mut() {
                            for g in v.iter_mut() {
                                *g *= -1.0;
                            }
                        }
                        compute_negate(tensor);
                    },
                    UnaryOp::Exp => {
                        for v in grad_values.iter_mut() {
                            for g in v.iter_mut() {
                                *g *= g.exp();
                            }
                        }
                        compute_exp(tensor);
                    },
                    UnaryOp::Reciprocal => {
                        for v in grad_values.iter_mut() {
                            for (g, x) in v.iter_mut().zip(&tensor.data) {
                                *g *= -1.0 / x.powi(2);
                            }
                        }
                        compute_reciprocal(tensor);
                    }
                };
                return (grad_ptrs, grad_values, tensor);
            },
            Node::BinaryOpNode { rhs, lhs, op } => {
                return rhs.eval();
            }
        }
    }
}

