#[derive(Debug, Clone)]
pub enum UnaryOp {
    Negate, Exp, Reciprocal
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add, Multiply, Pow
}
