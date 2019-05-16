# TensorPolynomialBases

[![Build Status](https://travis-ci.com/lssc-team/TensorPolynomialBases.jl.svg?branch=master)](https://travis-ci.com/lssc-team/TensorPolynomialBases.jl)
[![Codecov](https://codecov.io/gh/lssc-team/TensorPolynomialBases.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lssc-team/TensorPolynomialBases.jl)

The **TensorPolynomialBases** package provides a collection of different types prepresenting tensor-valued multivariate polynomial bases. It provides a common interface, called `TensorPolynomialBasis`, for all provided types and several concrete implementations. At the moment, only a concrete implementation, called `FixedPolynomialBasis`, which uses the [FixedPolynomials](https://github.com/JuliaAlgebra/FixedPolynomials.jl) package, is available. For representing the tensor values arising in the evaluation of tensor-valued polynomails, the user can either use the [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) or the [TensorValues](https://github.com/lssc-team/TensorValues.jl) packages.
