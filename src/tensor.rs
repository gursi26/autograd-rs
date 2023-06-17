#[derive(Eq, PartialEq, Hash)]
pub struct Value {
    pub data: f64,
    pub grad: Option<f64>,
    pub requires_grad: bool
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value {
            data,
            grad: None,
            requires_grad: false
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{{{}, requires_grad = {}}}", self.data, self.requires_grad)
    }
}
