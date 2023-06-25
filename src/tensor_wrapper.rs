use crate::tensor::{NodeTensor, VariableTensor};
use crate::variable_wrapper::Operable;
use std::ops::{Add, Mul};
use crate::variable_wrapper::*;
use rayon::prelude::*;


// Add implementations
impl <'a> Add for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let out_vec = self.values.par_iter_mut().zip(rhs.values.par_iter_mut()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        NodeTensor { values: out_vec }
    }
}

impl <'a> Add for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let out_vec = self.values.par_iter().zip(rhs.values.par_iter()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        NodeTensor { values: out_vec }
    }
}

impl <'a> Add<NodeTensor<'a>> for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.par_iter_mut()).map(|(rhs_x, lhs_x)| add(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Add<NodeTensor<'a>> for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.par_iter()).map(|(rhs_x, lhs_x)| add(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Add for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(self, mut rhs: Self) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.into_par_iter()).map(|(rhs_x, lhs_x)| add(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Add<&'a mut VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(mut self, rhs: &'a mut VariableTensor) -> Self::Output {
        self.values = self.values.into_par_iter().zip(rhs.values.par_iter_mut()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        self
    }
}

impl <'a> Add<&'a VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(mut self, rhs: &'a VariableTensor) -> Self::Output {
        self.values = self.values.into_par_iter().zip(rhs.values.par_iter()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        self
    }
}


// Mul implementations
impl <'a> Mul for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let out_vec = self.values.par_iter_mut().zip(rhs.values.par_iter_mut()).map(|(lhs_x, rhs_x)| mul(lhs_x, rhs_x)).collect();
        NodeTensor { values: out_vec }
    }
}

impl <'a> Mul for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let out_vec = self.values.par_iter().zip(rhs.values.par_iter()).map(|(lhs_x, rhs_x)| mul(lhs_x, rhs_x)).collect();
        NodeTensor { values: out_vec }
    }
}

impl <'a> Mul<NodeTensor<'a>> for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.par_iter_mut()).map(|(rhs_x, lhs_x)| mul(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Mul<NodeTensor<'a>> for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.par_iter()).map(|(rhs_x, lhs_x)| mul(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Mul for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(self, mut rhs: Self) -> Self::Output {
        rhs.values = rhs.values.into_par_iter().zip(self.values.into_par_iter()).map(|(rhs_x, lhs_x)| mul(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Mul<&'a mut VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(mut self, rhs: &'a mut VariableTensor) -> Self::Output {
        self.values = self.values.into_par_iter().zip(rhs.values.par_iter_mut()).map(|(lhs_x, rhs_x)| mul(lhs_x, rhs_x)).collect();
        self
    }
}

impl <'a> Mul<&'a VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(mut self, rhs: &'a VariableTensor) -> Self::Output {
        self.values = self.values.into_par_iter().zip(rhs.values.par_iter()).map(|(lhs_x, rhs_x)| mul(lhs_x, rhs_x)).collect();
        self
    }
}
