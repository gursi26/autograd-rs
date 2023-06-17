#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

mod tensor;
mod dcg;

fn main() {
    let val = tensor::Value::new(100.0);
    println!("{}", val);
}
