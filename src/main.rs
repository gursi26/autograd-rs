#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::println;

mod tensor;
mod graph;
mod ops;
mod compute;
mod wrapper;

fn main() {
    let mut t1 = parameter![1, 2, 3, 4, 5];
    let mut t2 = parameter![2, 2, 2, 2, 2];
    {
        let mut subtract_node = graph::Node::BinaryOpNode {
            rhs: Box::new(graph::Node::TensorNode { tensor: t1.clone(), source: &mut t1 }),
            lhs: Box::new(graph::Node::UnaryOpNode {
                value: Box::new(graph::Node::TensorNode { tensor: t2.clone(), source: &mut t2 }),
                op: ops::UnaryOp::Negate
            }),
            op: ops::BinaryOp::Add
        };
        let (mut grad_ptrs, mut grad_values, output) = subtract_node.eval();
        for (ptr, grad) in grad_ptrs.drain(..).zip(grad_values.drain(..)) {
            ptr.grad = Some(grad);
        }
        println!("Output: {:?}", output);
    }
    println!("T1: {:?}\nT2: {:?}", t1, t2);
}
