# API Reference

This page provides detailed documentation for all exported functions and types in ChenSignatures.jl.

## Primary Functions

These are the main user-facing functions for computing signatures and log-signatures:

```@docs
sig
logsig
prepare
rolling_sig
```

## Path Augmentations

Helpers for common path preprocessing patterns (time augmentation and lead–lag):

```@docs
time_augment
lead_lag
sig_time
sig_leadlag
logsig_time
logsig_leadlag
```

## Core Types

```@docs
Tensor
SignatureWorkspace
BasisCache
```

## Lower-Level API

Advanced users may need direct access to tensor-based computations:

```@docs
signature_path
```

## Index

```@index
Pages = ["api.md"]
```
