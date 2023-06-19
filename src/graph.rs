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
                let (mut rhs_grad_ptrs, mut rhs_grad_values, rhs_tensor) = rhs.eval();
                let (mut lhs_grad_ptrs, mut lhs_grad_values, lhs_tensor) = lhs.eval();
                // Computing output and grad_values
                match op {
                    BinaryOp::Add => {

                    },
                    BinaryOp::Pow => {},
                    BinaryOp::Multiply => {
                        for v in rhs_grad_values.iter_mut() {
                            for (g, lhs_x) in v.iter_mut().zip(&lhs_tensor.data) {
                                *g *= lhs_x;
                            }
                        }
                        for v in lhs_grad_values.iter_mut() {
                            for (g, rhs_x) in v.iter_mut().zip(&rhs_tensor.data) {
                                *g *= rhs_x;
                            }
                        }
                        compute_multiply(rhs_tensor, lhs_tensor);
                    }
                }

                // Merging grad_ptrs and grad_values from rhs and lhs
                let mut to_remove_idxs: Vec<usize> = Vec::new();
                for (i, (lhs_ptr, lhs_grad)) in lhs_grad_ptrs.iter().zip(&lhs_grad_values).enumerate() {
                    if rhs_grad_ptrs.contains(&lhs_ptr) {
                        for (rhs_grad_value, lhs_grad_value) in rhs_grad_values[i].iter_mut().zip(lhs_grad) {
                            *rhs_grad_value += *lhs_grad_value;
                        }
                    } else {
                        to_remove_idxs.push(i);
                    }
                }
                for idx in to_remove_idxs.iter() {
                    let removed_ptr = lhs_grad_ptrs.remove(*idx);
                    let removed_grad = lhs_grad_values.remove(*idx);
                    rhs_grad_ptrs.push(removed_ptr);
                    rhs_grad_values.push(removed_grad);
                }
                return (rhs_grad_ptrs, rhs_grad_values, rhs_tensor);
            }
        }
    }
}

