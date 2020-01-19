# Training

This repository enables the reproduction of the accuracy results reported in the article:
[3PXNet: Pruned-Permuted-Packed XNOR Networks for Edge Machine Learning](url)
The code is based on https://github.com/itayhubara/BinaryNet.pytorch

## Requirements

* Python 3.6, Numpy, pandas, bokeh
* PyTorch 0.4.0 or newer
* ONNX
* ONNX runtime
* NNEF-Tools (optional, will be installed by Makefile)

## MNIST

```bash
make prereq && make MNIST
```
Trains a small pruned binarized MLP on MNIST with permutation

## CIFAR10 & SVHN

```bash
make prereq && make CIFAR
```
Trains a large pruned binarized CNN on CIFAR10 with permutation
