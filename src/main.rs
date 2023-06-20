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
    let mut t = parameter![-0.1, 0.2, 0.8, 0.9, 3.14];
    let mut cnst = tensor![1, 1, 1, 1, 1];
    let sigmoid = reciprocal(add(&mut cnst, exp(negate(&mut t))));
    let sigmoid_output = eval(sigmoid);
    println!("Parameter: {:?}\nOutput: {:?}", t, sigmoid_output);
}
