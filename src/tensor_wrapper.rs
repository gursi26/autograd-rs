use crate::tensor::{NodeTensor, VariableTensor};
use crate::variable_wrapper::Operable;
use std::ops::{Add, Mul};
use crate::variable_wrapper::*;
use rayon::prelude::*;


// Add implementations
// impl <'a> Add for &'a mut VariableTensor {
//     type Output = NodeTensor<'a>;
//     fn add(self, rhs: Self) -> Self::Output {
//         let mut lhs = NodeTensor { values: Vec::new() };
//         for (lhs_x, rhs_x) in self.values.iter_mut().zip(&mut rhs.values) {
//             lhs.values.push(add(lhs_x, rhs_x));
//         }
//         lhs
//     }
// }

impl <'a> Add for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        // let mut lhs = NodeTensor { values: Vec::new() };
        // for (lhs_x, rhs_x) in self.values.iter_mut().zip(&mut rhs.values) {
        //     lhs.values.push(add(lhs_x, rhs_x));
        // }
        // lhs
        let out_vec = self.values.par_iter_mut().zip(rhs.values.par_iter_mut()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        // let out_vec = self.values.iter_mut().zip(rhs.values.iter_mut()).map(|(lhs_x, rhs_x)| add(lhs_x, rhs_x)).collect();
        NodeTensor { values: out_vec }
    }
}

impl <'a> Add for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter().zip(&rhs.values) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Add<NodeTensor<'a>> for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        // let mut lhs = NodeTensor { values: Vec::new() };
        // for (lhs_x, rhs_x) in self.values.iter_mut().zip(rhs.values.drain(..)) {
        //     lhs.values.push(add(lhs_x, rhs_x));
        // }
        // lhs
        rhs.values = rhs.values.into_par_iter().zip(self.values.par_iter_mut()).map(|(rhs_x, lhs_x)| add(lhs_x, rhs_x)).collect();
        rhs
    }
}

impl <'a> Add<NodeTensor<'a>> for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn add(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter().zip(rhs.values.drain(..)) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Add for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(mut self, mut rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(rhs.values.drain(..)) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Add<&'a mut VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(mut self, rhs: &'a mut VariableTensor) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(&mut rhs.values) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Add<&'a VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn add(mut self, rhs: &'a VariableTensor) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(&rhs.values) {
            lhs.values.push(add(lhs_x, rhs_x));
        }
        lhs
    }
}

// Mul implementations
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

impl <'a> Mul for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter().zip(&rhs.values) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul<NodeTensor<'a>> for &'a mut VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter_mut().zip(rhs.values.drain(..)) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul<NodeTensor<'a>> for &'a VariableTensor {
    type Output = NodeTensor<'a>;
    fn mul(self, mut rhs: NodeTensor<'a>) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.iter().zip(rhs.values.drain(..)) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(mut self, mut rhs: Self) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(rhs.values.drain(..)) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul<&'a mut VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(mut self, rhs: &'a mut VariableTensor) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(&mut rhs.values) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}

impl <'a> Mul<&'a VariableTensor> for NodeTensor<'a> {
    type Output = NodeTensor<'a>;
    fn mul(mut self, rhs: &'a VariableTensor) -> Self::Output {
        let mut lhs = NodeTensor { values: Vec::new() };
        for (lhs_x, rhs_x) in self.values.drain(..).zip(&rhs.values) {
            lhs.values.push(mul(lhs_x, rhs_x));
        }
        lhs
    }
}
