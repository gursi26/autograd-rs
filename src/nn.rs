use crate::graph::Node;
use crate::variable::*;
use crate::variable_wrapper::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

pub trait OperableVec {}
impl OperableVec for &mut Vec<Variable> {}
impl OperableVec for &mut Vec<Vec<Variable>> {}
impl OperableVec for &Vec<Variable> {}
impl<'a> OperableVec for Vec<Node<'a>> {}

pub enum Layer {
    Linear(Vec<Vec<Variable>>, Vec<Variable>),
    Sigmoid,
}

impl Layer {
    pub fn linear(in_shape: usize, out_shape: usize) -> Layer {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut weights = Vec::new();

        for _ in 0..out_shape {
            weights.push(
                (0..in_shape)
                    .into_par_iter()
                    .map(|_| parameter!(normal.sample(&mut rand::thread_rng())))
                    .collect(),
            );
        }

        let biases = (0..out_shape)
            .into_par_iter()
            .map(|_| parameter!(normal.sample(&mut rand::thread_rng())))
            .collect();

        Layer::Linear(weights, biases)
    }
}

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub fn new(layers: Vec<Layer>) -> Sequential {
        Sequential { layers }
    }

    pub fn forward<'a>(&mut self, xvals: &Vec<f64>) -> Vec<Node<'a>> {
        let xvals: Vec<Variable> = xvals.into_iter().map(|x| Variable::new(*x)).collect();
        for layer in self.layers.iter_mut() {
            match layer {
                Layer::Linear(w, b) => {
                    
                }
                Layer::Sigmoid => {}
            };
        }
        vec![negate(1.0)]
    }
}

#[macro_export]
macro_rules! sequential {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            Sequential::new(temp_vec)
        }
    };
}
