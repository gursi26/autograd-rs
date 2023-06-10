use std::fmt;
use crate::dcg::DCGNodeTrait;

pub struct Tensor {
    values: Vec<f64>,
    grads: Option<Vec<f64>>,
    pub requires_grad: bool
}

impl DCGNodeTrait for Tensor {}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            values: Vec::new(),
            grads: None,
            requires_grad: false
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{{:?}, requires_grad = {}}}", self.values, self.requires_grad)
    }
}

impl Tensor {
    pub fn new() -> Tensor {
        Tensor {..Default::default()}
    }

    pub fn from(values: &[f64]) -> Tensor {
        Tensor {values: Vec::from(values), ..Default::default()}
    }

    pub fn from_vec(values: Vec<f64>) -> Tensor {
        Tensor {values, ..Default::default()}
    }
}

#[macro_export]
macro_rules! tensor {
    ($($x: expr),*) => {{
        let mut vector = Vec::new();
        $(vector.push($x as f64);)*
        Tensor::from_vec(vector)
    }}
}
