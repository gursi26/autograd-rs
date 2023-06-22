#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod tensor;
mod graph;
mod ops;
mod compute;
mod wrapper;

use wrapper::*;

fn main() {
    let mut weight = parameter![1.0];
    let mut bias = parameter![1.0];
    let mut xvals = tensor![1, 2, 3, 4, 5];
    let mut yvals = tensor![5, 8, 11, 14, 17];

    let output = add(mul(&mut weight, &mut xvals), &mut bias);
    let loss = mul(sum(pow(add(&mut yvals, negate(output)), 2.0)), reciprocal(5.0));
    let output = eval(loss);
    println!("{:?}", output);
    println!("Weight: {:?}\nBias: {:?}", weight, bias);
}
