use crate::tensor::Value;
use std::collections::HashMap;

#[derive(Eq, PartialEq, Hash)]
enum BinaryOp {
    Add, Multiply, Pow
}

#[derive(Eq, PartialEq, Hash)]
enum UnaryOp {
    Negate, Exp, Log, Reciprocal
}

#[derive(Eq, PartialEq, Hash)]
enum DCGNode<'a> {
    TensorNode {
        value: Box<Value>,
        source: &'a mut Value
    },
    BinaryOpNode {
        op: BinaryOp,
        rhs: Box<DCGNode<'a>>,
        lhs: Box<DCGNode<'a>>,
        grad_dict: HashMap<&'a Box<DCGNode<'a>>, f64>
    },
    UnaryOpNode {
        op: UnaryOp,
        value: Box<DCGNode<'a>>,
        grad_dict: HashMap<&'a Box<DCGNode<'a>>, f64>
    }
}

impl<'a> DCGNode<'a> {
    pub fn evaluate(&self) -> HashMap<&'a Box<DCGNode<'a>>, f64> {
        match self {
            Self::TensorNode { value, source } => {
                let mut grad_dict: HashMap<&'a Box<DCGNode<'a>>, f64> = HashMap::new(); 
                if value.requires_grad {
                    grad_dict.insert(Self, 1.0);
                }
                grad_dict
            }
        }


    }
}
