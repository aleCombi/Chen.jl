# Chen

**Chen** is a high-performance Julia library for computing path signatures and log-signatures, foundational tools in Rough Path Theory and modern time-series analysis.

[![Build Status](https://github.com/aleCombi/Chen.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/aleCombi/Chen.jl/actions/workflows/CI.yml?query=branch%3Amaster)

## 📐 Overview

Chen provides efficient tools for calculating the signature of a path—a sequence of iterated integrals that captures the geometric and algebraic properties of sequential data. It is designed to be a faster, Julia-native alternative to the Python library `iisignature`.

## ✅ Key Features

- **Path Signatures**: Efficient computation of truncated signatures for vector-valued paths.
- **Log-Signatures**: Computation of log-signatures projected onto the Lyndon basis for minimal representation.
- **Tensor Algebra**: Robust support for both **Dense** and **Sparse** tensors (`SparseTensor`).
- **Algebraic Operations**: Implementation of shuffle products, tensor exponentials, logarithms, and resolvents.

## 🚀 Highlights

- **Performance**: Optimized kernels utilizing `LoopVectorization.jl` and `StaticArrays.jl` for high-speed computation.
- **Sparse Support**: Efficient handling of high-dimensional algebraic structures via sparse representations.
- **Validated**: core algorithms are validated against the industry-standard `iisignature` library.

## 📦 Dependencies

Chen is built on a minimal set of high-performance packages:

- `StaticArrays.jl` - For efficient small-vector operations.
- `LoopVectorization.jl` - For SIMD-optimized inner loops.
- `LinearAlgebra` - For tensor basis projections.
