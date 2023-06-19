use crate::tensor::Tensor;

pub fn compute_negate(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x *= -1.0;
    }
}

pub fn compute_exp(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x = x.exp();
    }
}

pub fn compute_reciprocal(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x = 1.0 / *x;
    }
}

pub fn compute_multiply(to_mutate: &mut Tensor, other_tensor: &Tensor) {
    for (x, y) in to_mutate.data.iter_mut().zip(&other_tensor.data) {
        *x *= y;
    }
}

pub fn compute_merged_grads<'a>(
    rhs_ptrs: &mut Vec<&'a mut Tensor>,
    mut lhs_ptrs: Vec<&'a mut Tensor>,
    rhs_grads: &mut Vec<Vec<f64>>,
    mut lhs_grads: Vec<Vec<f64>>
) {
    for (ptr, grad) in lhs_ptrs.drain(..).zip(lhs_grads.drain(..)) {
        match rhs_ptrs.iter().position(|p| *p == ptr) {
            Some(idx) => {
                for (rhs_single_grad, lhs_single_grad) in rhs_grads[idx].iter_mut().zip(&grad) {
                    *rhs_single_grad += lhs_single_grad;
                }
            }
            None => {
                rhs_ptrs.push(ptr);
                rhs_grads.push(grad);
            }
        };
    }
}

