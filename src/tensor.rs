#[derive(Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub length: usize,
    pub grad: Option<Vec<f64>>,
    pub requires_grad: bool
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            data: Vec::new(),
            length: 0,
            grad: None,
            requires_grad: false
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            length: self.length,
            grad: self.grad.clone(),
            requires_grad: self.requires_grad
        }
    }
}


impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{{{:?}, requires_grad = {}}}", self.data, self.requires_grad)
    }
}

impl Tensor {
    pub fn new() -> Tensor {
        Tensor {..Default::default()}
    }

    pub fn from(data: &[f64]) -> Tensor {
        Tensor {data: Vec::from(data), length: data.len(), ..Default::default()}
    }

    pub fn from_vec(data: Vec<f64>) -> Tensor {
        Tensor {length: data.len(), data, ..Default::default()}
    }

    pub fn parameter(data: Vec<f64>) -> Tensor {
        Tensor {length: data.len(), data, requires_grad: true, ..Default::default()}
    }
}


#[macro_export]
macro_rules! tensor {
    ($($x: expr),*) => {{
        let mut vector = Vec::new();
        $(vector.push($x as f64);)*
        tensor::Tensor::from_vec(vector)
    }}
}

#[macro_export]
macro_rules! parameter {
    ($($x: expr),*) => {{
        let mut vector = Vec::new();
        $(vector.push($x as f64);)*
        tensor::Tensor::parameter(vector)
    }}
}
