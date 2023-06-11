#![allow(dead_code)]
#![allow(unused_imports)]

mod tensor;
mod dcg;
mod ops;
mod compute;


fn main() {
    let t1 = Box::new(dcg::DCGNode::TensorNode(vec![1.0, 2.0]));
    let t2 = Box::new(dcg::DCGNode::TensorNode(vec![3.0, 4.0]));
    let root = dcg::DCGNode::BinaryNode { op: ops::BinaryOp::Multiply, lhs: t1, rhs: t2 };
    let root1 = dcg::DCGNode::UnaryNode { op: ops::UnaryOp::Negate, value: Box::new(root) };
    let output = root1.eval();
    println!("{:?}", output);
}
