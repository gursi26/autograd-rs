use crate::wrapper::*;
use crate::variable::Variable;
use std::ops::{Add, Mul};
use crate::graph::Node;

#[derive(Clone, Debug)]
pub struct VariableTensor {
    pub values: Vec<Variable>
}

#[derive(Debug)]
pub struct NodeTensor<'a> {
    pub values: Vec<Node<'a>>
}

pub trait OperableTensor<'a> {
    fn make_operable(self) -> NodeTensor<'a>;
}

impl <'a> OperableTensor<'a> for &'a VariableTensor {
    fn make_operable(self) -> NodeTensor<'a> {
        let mut new_tensor = NodeTensor{values: Vec::new()};
        for value in self.values.iter() {
            new_tensor.values.push(value.make_operable());
        }
        new_tensor
    }
}

impl <'a> OperableTensor<'a> for NodeTensor<'a> {
    fn make_operable(self) -> NodeTensor<'a> {
        self
    }
}

impl <'a> Add for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter_mut().zip(&mut rhs.values) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter_mut().zip(&mut rhs.values) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}
