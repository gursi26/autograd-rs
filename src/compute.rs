use crate::ops::{UnaryOp, BinaryOp};

pub fn compute_unary(op: UnaryOp, v: Vec<f64>) -> Vec<f64> {
    match op {
        UnaryOp::Exp => compute_unary_exp(v),
        UnaryOp::Log => compute_unary_log(v),
        UnaryOp::Negate => compute_unary_negate(v),
        UnaryOp::Reciprocal => compute_unary_reciprocal(v)
    }
}

fn compute_unary_exp(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|x| x.exp()).collect()
}

fn compute_unary_negate(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|x| -x).collect()
}

fn compute_unary_reciprocal(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|x| 1.0 / x).collect()
}

fn compute_unary_log(v: Vec<f64>) -> Vec<f64> {
    v.into_iter().map(|x| x.ln()).collect()
}

pub fn compute_binary(op: BinaryOp, lhs: Vec<f64>, rhs: Vec<f64>) -> Vec<f64> {
    match op {
        BinaryOp::Add => compute_binary_add(lhs, rhs),
        BinaryOp::Pow => compute_binary_pow(lhs, rhs),
        BinaryOp::Multiply => compute_binary_mul(lhs, rhs)
    }
}

fn compute_binary_add(v1: Vec<f64>, v2: Vec<f64>) -> Vec<f64> {
    let mut return_vec = vec![0.0; v1.len()];
    for (i, (x1, x2)) in v1.iter().zip(&v2).enumerate() {
        return_vec[i] = x1 + x2;
    }
    return_vec
}

fn compute_binary_pow(v1: Vec<f64>, v2: Vec<f64>) -> Vec<f64> {
    let mut return_vec = vec![0.0; v1.len()];
    for (i, (x1, x2)) in v1.iter().zip(&v2).enumerate() {
        return_vec[i] = ((*x1) as f64).powf((*x2) as f64);
    }
    return_vec
}

fn compute_binary_mul(v1: Vec<f64>, v2: Vec<f64>) -> Vec<f64> {
    let mut return_vec = vec![0.0; v1.len()];
    for (i, (x1, x2)) in v1.iter().zip(&v2).enumerate() {
        return_vec[i] = x1 * x2;
    }
    return_vec
}
