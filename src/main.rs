#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod compute;
mod graph;
mod nn;
mod ops;
mod variable;
mod variable_wrapper;

use compute::matmul;
use nn::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use variable_wrapper::*;

fn main() {
    let layer = Layer::linear(100, 200);
    let mut x = Vec::new();
    for _ in 0..100 {
        x.push(add(1.0, 0.0))
    }
    if let Layer::Linear(mut w, b) = layer {
        let out = matmul(&mut w, x);
        let out: Vec<Variable> = out.into_iter().map(|n| eval(n)).collect();
        println!("{:?}", out);
    }
}
