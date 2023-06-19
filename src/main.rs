#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::println;

mod tensor;
mod graph;
mod ops;
mod compute;

fn main() {
    let mut t1 = tensor![1, 2, 3, 4, 5];
    t1.requires_grad = true;
    let mut t2 = tensor![10, 20, 10, 20, 10];
    t2.requires_grad = true;

    {
        let mut mul_node = graph::Node::BinaryOpNode {
            rhs: Box::new(graph::Node::TensorNode { tensor: t1.clone(), source: &mut t1 }),
            lhs: Box::new(graph::Node::TensorNode { tensor: t2.clone(), source: &mut t2 }),
            op: ops::BinaryOp::Multiply
        };
        let (mut grad_ptrs, mut grad_values, output_tensor) = mul_node.eval();
        let mut i = (grad_ptrs.len() - 1) as i32;
        for ptr in grad_ptrs.iter_mut().rev() {
            (*ptr).grad = Some(grad_values.remove(i as usize));
            i -= 1;
        }
        println!("{:?}", output_tensor);
    }
    println!("{:?}\n{:?}", t1, t2);
}
