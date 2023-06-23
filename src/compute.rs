use crate::variable::Variable;

pub fn update_grads(grad_values: &mut Vec<f64>, new_grad: f64) {
    for grad in grad_values.iter_mut() {
        *grad *= new_grad;
    }
}

pub fn compute_merged_grads<'a> (
    mut rhs_ptrs: Vec<&'a mut Variable>,
    lhs_ptrs: &mut Vec<&'a mut Variable>,
    mut rhs_grads: Vec<f64>,
    lhs_grads: &mut Vec<f64>,
) {
    for (ptr, grad) in rhs_ptrs.drain(..).zip(rhs_grads.drain(..)) {
        match lhs_ptrs.iter().position(|p| *p == ptr) {
            Some(idx) => {
                lhs_grads[idx] += grad;
            }
            None => {
                lhs_ptrs.push(ptr);
                lhs_grads.push(grad);
            }
        };
    }
}
