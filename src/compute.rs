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

