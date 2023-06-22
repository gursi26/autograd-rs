#[derive(Debug)]
pub enum UnaryOp {
    Negate, Exp, Reciprocal, Sum
}

#[derive(Debug)]
pub enum BinaryOp {
    Add, Multiply, Pow
}
