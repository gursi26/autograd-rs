#[derive(Debug, PartialEq, Clone)]
pub struct Variable {
    pub value: f64,
    pub requires_grad: bool,
    pub grad: Option<f64>,
}

impl Default for Variable {
    fn default() -> Self {
        Variable {
            value: 0.0,
            requires_grad: false,
            grad: None,
        }
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{{{ }, requires_grad = { }}}",
            self.value, self.requires_grad
        )
    }
}

impl Variable {
    pub fn new(value: f64) -> Variable {
        Variable {
            value,
            requires_grad: false,
            ..Default::default()
        }
    }

    pub fn parameter(value: f64) -> Variable {
        Variable {
            value,
            requires_grad: true,
            ..Default::default()
        }
    }
}

#[macro_export]
macro_rules! var {
    ($a:expr) => {{
        Variable::new($a as f64)
    }};
}

#[macro_export]
macro_rules! parameter {
    ($a:expr) => {{
        Variable::parameter($a as f64)
    }};
}

pub use parameter;
pub use var;
