use crate::tensor::Tensor;

pub fn compute_negate(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x *= -1.0;
    }
}

pub fn compute_negate_grad(grad_values: &mut Vec<Vec<f64>>) {
    for grad_vec in grad_values.iter_mut() {
        for grad in grad_vec.iter_mut() {
            *grad *= -1.0;
        }
    }
}

pub fn compute_exp(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x = x.exp();
    }
}

pub fn compute_exp_grad(tensor: &Tensor, grad_values: &mut Vec<Vec<f64>>) { 
    for grad_vec in grad_values.iter_mut() {
        for (grad, xval) in grad_vec.iter_mut().zip(&tensor.data) {
            *grad *= xval.exp();
        }
    }
}

pub fn compute_reciprocal(tensor: &mut Tensor) {
    for x in tensor.data.iter_mut() {
        *x = 1.0 / *x;
    }
}

pub fn compute_reciprocal_grad(tensor: &Tensor, grad_values: &mut Vec<Vec<f64>>) { 
    for grad_vec in grad_values.iter_mut() {
        for (grad, xval) in grad_vec.iter_mut().zip(&tensor.data) {
            *grad *= -1.0 / xval.powi(2);
        }
    }
}

pub fn compute_sum(tensor: &mut Tensor) {
    while tensor.data.len() > 1 {
        tensor.data[0] += tensor.data.pop().unwrap();
    }
    tensor.length = 1;
}

pub fn compute_sum_grad(grad_values: &mut Vec<Vec<f64>>) {
    for tensor in grad_values.iter_mut() {
        while tensor.len() > 1 {
            tensor[0] += tensor.pop().unwrap();
        }
    }
}

pub fn compute_multiply(to_mutate: &mut Tensor, other_tensor: &Tensor) {
    assert_eq!(
        to_mutate.data.len(), other_tensor.data.len(),
        "Attempted to multiply Tensors with unequal length."
    );
    for (x, y) in to_mutate.data.iter_mut().zip(&other_tensor.data) {
        *x *= y;
    }
}

pub fn compute_multiply_grad(
    rhs_tensor: &Tensor, lhs_tensor: &Tensor,
    rhs_grads: &mut Vec<Vec<f64>>, lhs_grads: &mut Vec<Vec<f64>>
) {
    for grad_vec in rhs_grads.iter_mut() {
        for (grad, lhs_x) in grad_vec.iter_mut().zip(&lhs_tensor.data) {
            *grad *= lhs_x;
        }
    }
    for grad_vec in lhs_grads.iter_mut() {
        for (grad, rhs_x) in grad_vec.iter_mut().zip(&rhs_tensor.data) {
            *grad *= rhs_x;
        }
    }
}

pub fn compute_add(to_mutate: &mut Tensor, other_tensor: &Tensor) {
    assert_eq!(
        to_mutate.data.len(), other_tensor.data.len(),
        "Attempted to add Tensors with unequal length."
    );
    for (x, y) in to_mutate.data.iter_mut().zip(&other_tensor.data) {
        *x += y;
    }
}

pub fn compute_pow(to_mutate: &mut Tensor, other_tensor: &Tensor) {
    assert_eq!(
        to_mutate.data.len(), other_tensor.data.len(),
        "Attempted to pow Tensors with unequal length."
    );
    for (x, y) in to_mutate.data.iter_mut().zip(&other_tensor.data) {
        *x = x.powf(*y);
    }
}

pub fn compute_pow_grad(
    rhs_tensor: &Tensor, lhs_tensor: &Tensor,
    rhs_grads: &mut Vec<Vec<f64>>, lhs_grads: &mut Vec<Vec<f64>>
) {
    for grad_vec in rhs_grads.iter_mut() {
        for ((grad, lhs_x), rhs_x) in grad_vec.iter_mut().zip(&lhs_tensor.data).zip(&rhs_tensor.data) {
            *grad *= lhs_x.ln() * lhs_x.powf(*rhs_x);
        }
    }
    for grad_vec in lhs_grads.iter_mut() {
        for ((grad, rhs_x), lhs_x) in grad_vec.iter_mut().zip(&rhs_tensor.data).zip(&lhs_tensor.data) {
            *grad *= rhs_x * lhs_x.powf(rhs_x - 1.0);
        }
    }
}


pub fn compute_merged_grads<'a>(
    lhs_ptrs: &mut Vec<&'a mut Tensor>,
    mut rhs_ptrs: Vec<&'a mut Tensor>,
    lhs_grads: &mut Vec<Vec<f64>>,
    mut rhs_grads: Vec<Vec<f64>>
) {
    for (ptr, grad) in rhs_ptrs.drain(..).zip(rhs_grads.drain(..)) {
        match lhs_ptrs.iter().position(|p| *p == ptr) {
            Some(idx) => {
                for (lhs_single_grad, rhs_single_grad) in lhs_grads[idx].iter_mut().zip(&grad) {
                    *lhs_single_grad += rhs_single_grad;
                }
            }
            None => {
                lhs_ptrs.push(ptr);
                lhs_grads.push(grad);
            }
        };
    }
}

pub fn compute_equalized_length(
    t1: &mut Tensor,
    t2: &mut Tensor,
    t1_grads: &mut Vec<Vec<f64>>,
    t2_grads: &mut Vec<Vec<f64>>
) {
    let to_mutate: &mut Tensor;
    let to_mutate_grad: &mut Vec<Vec<f64>>;
    let other_tensor: &mut Tensor;
    if t1.length == 1 && t2.length > 1 {
        to_mutate = t1;
        to_mutate_grad = t1_grads;
        other_tensor = t2;
    } else if t2.length == 1 && t1.length > 1 {
        to_mutate = t2;
        to_mutate_grad = t2_grads;
        other_tensor = t1;
    } else {
        return;
    }

    while to_mutate.data.len() < other_tensor.length {
        to_mutate.data.push(to_mutate.data[0]);
    }
    for grad_vec in to_mutate_grad.iter_mut() {
        while grad_vec.len() < other_tensor.length {
            grad_vec.push(grad_vec[0]);
        }
    }

    to_mutate.length = to_mutate.data.len();
}
