use crate::variable_wrapper::*;
use crate::variable::Variable;
use std::ops::{Add, Mul};
use crate::graph::Node;
use std::fmt;
use rayon::prelude::*;

#[derive(Clone)]
pub struct VariableTensor {
    pub values: Vec<Variable>
}

#[derive(Debug)]
pub struct NodeTensor<'a> {
    pub values: Vec<Node<'a>>
}

impl VariableTensor {
    pub fn new(values: Vec<f64>, requires_grad: bool) -> VariableTensor {
        if requires_grad {
            return VariableTensor { values: values.iter().map(|x| Variable::parameter(*x)).collect() };
        }
        VariableTensor { values: values.iter().map(|x| Variable::new(*x)).collect() }
    }
}

impl fmt::Display for VariableTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{[")?;
        for i in 0..(self.values.len() - 1) {
            write!(f, "{}, ", self.values[i].value)?;
        }
        write!(f, "{}], requires_grad = {}}}", self.values.last().unwrap().value, self.values[0].requires_grad)?;
        Ok(())
    }
}

impl fmt::Debug for VariableTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{[")?;
        for i in 0..(self.values.len() - 1) {
            write!(f, "{}, ", self.values[i].value)?;
        }
        write!(f, "{}], grads = [", self.values.last().unwrap().value)?;
        for i in 0..(self.values.len() - 1) {
            match self.values[i].grad {
                Some(g) => write!(f, "{}, ", g),
                None => write!(f, "None, ")
            }?;
        }
        match self.values.last().unwrap().grad {
            Some(g) => write!(f, "{}], requires_grad = {}}}", g, self.values[0].requires_grad),
            None => write!(f, "None], requires_grad = {}}}", self.values[0].requires_grad)
        }?;
        Ok(())
    }
}

impl <'a> NodeTensor<'a> {
    pub fn eval(self) -> VariableTensor {
        VariableTensor {
            values: self.values.into_par_iter().map(|x| eval(x)).collect()
        }
    }
}

#[macro_export]
macro_rules! tensor {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x as f64);
            )*
            VariableTensor::new(temp_vec, false)
        }
    };
}

#[macro_export]
macro_rules! parameter_tensor {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x as f64);
            )*
            VariableTensor::new(temp_vec, true)
        }
    };
}

