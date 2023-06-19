#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod tensor;
mod graph;
mod ops;
mod compute;

fn main() {
    let mut t1 = tensor![1, 2, 3, 4, 5];
    t1.requires_grad = true;
    let mut negate_node = graph::Node::UnaryOpNode {
        value: Box::new(graph::Node::TensorNode { tensor: t1.clone(), source: &mut t1 }),
        op: ops::UnaryOp::Reciprocal
    };
    let (grad_ptrs, grad_values, output_tensor) = negate_node.eval();
    println!("{:?}\n {:?}\n {:?}", output_tensor, grad_ptrs, grad_values);
}
