#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

mod variable;
mod graph;
mod ops;
mod compute;
mod wrapper;
mod tensor;

use variable::Variable;
use tensor::VariableTensor;
use wrapper::*;

fn main() {
    let mut a = VariableTensor{ values: vec![parameter!(1), parameter!(2), parameter!(3)]};
    let mut b = VariableTensor{ values: vec![parameter!(4), parameter!(5), parameter!(6)]};
    let mut add_node = &mut a * &mut b;
    let mut output_values = Vec::new();
    for val in add_node.values.drain(..) {
        output_values.push(eval(val));
    }
    println!("{:?}", output_values);
    println!("{:?}", a);
    println!("{:?}", b);
}
