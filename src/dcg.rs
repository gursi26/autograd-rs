use crate::ops::ElementaryOp;

pub trait DCGNodeTrait {}

struct DCGNode<'a, T: DCGNodeTrait> {
    root: ElementaryOp,
    left: &'a T,
    right: &'a T
}

impl<'a, T: DCGNodeTrait> DCGNodeTrait for DCGNode<'a, T> {}
