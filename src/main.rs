#![allow(dead_code)]

mod tensor;
mod dcg;
mod ops;
use tensor::Tensor;

fn main() {
    let a = Tensor::from(&[1.1, 2.2]);
    let mut b = tensor![1, 2, 3, 4, 5];
    b.requires_grad = true;
    let c = Tensor::new();
    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
}
