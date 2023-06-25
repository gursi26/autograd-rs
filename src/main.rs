#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod variable;
mod graph;
mod ops;
mod compute;
mod variable_wrapper;
mod tensor;
mod tensor_wrapper;

use variable::Variable;
use tensor::*;
use variable_wrapper::*;
use rayon;

fn main() {
    let mut i = 0;
    let n = 1_000_000;
    loop {
        let mut a = VariableTensor::new(vec![10.0; n], true);
        let mut b = VariableTensor::new(vec![100.0; n], true);
        let mut c = VariableTensor::new(vec![27.0; n], true);
        let mut d = VariableTensor::new(vec![93.0; n], true);

        let out = (&mut a * &mut b + &mut c + &mut d).eval();
        println!("Iteration {}", i);
        i += 1;
    }
}
