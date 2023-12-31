use crate::{variable::Variable, graph::Node};
use rayon::prelude::*;
use crate::variable_wrapper::*;

pub fn update_grads(grad_values: &mut Vec<f64>, new_grad: f64) {
    for grad in grad_values.iter_mut() {
        *grad *= new_grad;
    }
    // grad_values.par_iter_mut().for_each(|x| *x *= new_grad);
}

pub fn compute_merged_grads<'a>(
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

pub fn matmul<'a>(w: &'a mut Vec<Vec<Variable>>, x: Vec<Node<'a>>) -> Vec<Node<'a>> {
    w.par_iter_mut()
        .map(|w_vec| sum(dot(w_vec, x.clone())))
        .collect()
}

pub fn dot<'a>(w: &'a mut Vec<Variable>, mut x: Vec<Node<'a>>) -> Vec<Node<'a>> {
    assert_eq!(w.len(), x.len());
    w.par_iter_mut()
        .zip(x.par_drain(..))
        .map(|(wval, xval)| mul(wval, xval))
        .collect()
}

pub fn sum(mut v: Vec<Node>) -> Node {
    let mut s = add(0, 0);
    for val in v.drain(..) {
        s = add(s, val);
    }
    s
}
