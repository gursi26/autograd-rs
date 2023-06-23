#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod variable;
mod graph;
mod ops;
mod compute;
mod wrapper;

use variable::Variable;
use wrapper::*;

fn main() {
    let mut w = parameter!(1);
    let mut b = parameter!(4);
    let xval = var![70];
    let yval = var![421];

    let yhat = add(mul(&mut w, &xval), &mut b);
    let loss = eval(pow(add(yhat, negate(&yval)), 2));

    println!("Loss : {}", loss);
    println!("Weight : {:?}", w);
    println!("Bias : {:?}", b);
}
