pub struct Tensor {
    values: Vec<f64>,
    grads: Option<Vec<f64>>,
    requires_grad: bool
}
