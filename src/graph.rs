use crate::tensor::Tensor;
use crate::ops::{UnaryOp, BinaryOp};
use crate::compute::*;

#[derive(Debug)]
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
                        compute_negate_grad(&mut grad_values);
                        compute_negate(tensor);
                    },
                    UnaryOp::Exp => {
                        compute_exp_grad(&*tensor, &mut grad_values);
                        compute_exp(tensor);
                    },
                    UnaryOp::Reciprocal => {
                        compute_reciprocal_grad(&*tensor, &mut grad_values);
                        compute_reciprocal(tensor);
                    },
                    UnaryOp::Sum => {
                        compute_sum_grad(&mut grad_values);
                        compute_sum(tensor)
                    }
                };
                return (grad_ptrs, grad_values, tensor);
            },
            Node::BinaryOpNode { rhs, lhs, op } => {
                let (rhs_grad_ptrs, mut rhs_grad_values, rhs_tensor) = rhs.eval();
                let (mut lhs_grad_ptrs, mut lhs_grad_values, lhs_tensor) = lhs.eval();

                compute_equalized_length(rhs_tensor, lhs_tensor, &mut rhs_grad_values, &mut lhs_grad_values);

                match op {
                    BinaryOp::Add => {
                        compute_add(lhs_tensor, rhs_tensor);
                    },
                    BinaryOp::Pow => {
                        compute_pow_grad(&*rhs_tensor, &*lhs_tensor, &mut rhs_grad_values, &mut lhs_grad_values);
                        compute_pow(lhs_tensor, rhs_tensor)
                    },
                    BinaryOp::Multiply => {
                        compute_multiply_grad(&*rhs_tensor, &*lhs_tensor, &mut rhs_grad_values, &mut lhs_grad_values);
                        compute_multiply(lhs_tensor, rhs_tensor);
                    }
                }

                compute_merged_grads(
                    &mut lhs_grad_ptrs, rhs_grad_ptrs,
                    &mut lhs_grad_values, rhs_grad_values
                );
                return (lhs_grad_ptrs, lhs_grad_values, lhs_tensor);
            }
        }
    }
}

