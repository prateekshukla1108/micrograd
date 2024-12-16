# Micrograd C: Automatic Differentiation in C

## Overview


![micrograd](micrograd.png)

Micrograd C is a lightweight implementation of automatic differentiation (autograd) in C, inspired by the Python micrograd library by Andrej Karpathy. This project provides a simple computational graph with backwpropagation support, allowing for gradient computation in basic mathematical operations.

## Features

- Automatic gradient computation
- Support for basic operations:
  - Addition
  - Multiplication
  - ReLU activation
- Recursive backpropagation
- Memory management for computational graphs
- Interactive demonstration of autograd concepts

## Prerequisites

- GCC or Clang compiler
- Standard C library
- Math library (`-lm` when compiling)

## Compilation

Compile the program using:

```bash
gcc -o micrograd micrograd.c -lm
```

## Usage

Run the program and follow the interactive prompts:

```bash
./micrograd
```

The program will guide you through:
1. Entering input values
2. Selecting a computation type
3. Displaying computational results
4. Showing gradients
5. Printing the computational graph

### Computation Types

1. Multiplication + Addition + ReLU
2. Only Multiplication
3. Only Addition

## Computational Graph

The implementation creates a computational graph that tracks:
- Node values
- Gradients
- Operation types
- Gradient requirements

## Key Components

- `Value` struct: Core data structure for computational graph nodes
- Backpropagation methods for different operations
- Memory management functions
- Computational graph visualization

## Example Output

```
Welcome to Micrograd C - An Autograd Demonstration
Enter first input value: 2
Enter second input value: 3
Enter bias value: 1

Select computation type:
1. Multiplication + Addition + ReLU
2. Only Multiplication
3. Only Addition
Enter your choice (1-3): 1

Computational Results
--------------------
Output: 7.00

Gradients
---------
Input1 gradient: 3.00
Input2 gradient: 2.00
Bias gradient: 1.00
```

## Limitations

- Currently supports a limited set of operations
- Minimal error handling
- Designed for educational purposes


## Acknowledgments

Inspired by the original micrograd implementation by Karpathy in Python.
