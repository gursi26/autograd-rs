# autograd-rs

A simple autograd implementation in Rust.
Features include:
- **Zero intermediate states** - No intermediate output tensors are created while evaluating large functions.
- **Single pass** - Forwards and backwards passes are not separate. Function outputs and gradients are calculated in a single pass. This comes at the cost of some flexibility.
- **Minimal Tensor duplication** - Tensors are only duplicated once when passed as inputs to a function. All operations mutate existing tensors.

This project is still a work in progress.
