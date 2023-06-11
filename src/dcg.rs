use crate::ops::{UnaryOp, BinaryOp};
use crate::compute;
use crate::tensor;


pub enum DCGNode {
    TensorNode(Vec<f64>),
    UnaryNode {
        op: UnaryOp,
        value: Box<DCGNode>
    },
    BinaryNode {
        op: BinaryOp,
        lhs: Box<DCGNode>,
        rhs: Box<DCGNode>
    }
}


impl DCGNode {
    pub fn eval(self) -> Vec<f64> {
        match self {
            DCGNode::TensorNode(v) => v,
            DCGNode::UnaryNode { op, value } => {
                compute::compute_unary(op, value.eval())
            }
            DCGNode::BinaryNode { op, lhs, rhs } => {
                compute::compute_binary(op, lhs.eval(), rhs.eval())
            }
        }
    }
}
