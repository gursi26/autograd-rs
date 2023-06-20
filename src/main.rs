#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::println;

use wrapper::Operable;

mod tensor;
mod graph;
mod ops;
mod compute;
mod wrapper;

use wrapper::*;

fn main() {
    let mut i = 0;
    loop {
        let mut t = tensor::Tensor::rand(100_000);
        let mut cnst = tensor::Tensor::from_vec(vec![1.0; 100_000]);
        t.requires_grad = true;
        let sigmoid_output = eval(reciprocal(add(&mut cnst, exp(negate(&mut t)))));
        println!("Completed {}", i);
        i += 1;
    }
}
