# Factorization-based segmentation Python implementation

This is a Python implementation of the factorization-based segmentation algorithm, which fast segments textured images. The algorithm is described in 

**J. Yuan, D. L. Wang, and A. M. Cheriyadat. Factorization-based texture segmentation. IEEE Transactions on Image Processing, 2015.**

[Here](https://sites.google.com/site/factorizationsegmentation/) is a brief introduction of the algorithm. [Here](https://medium.com/@jiangye07/fast-local-histogram-computation-using-numpy-array-operations-d96eda02d3c) is an explanation of computing local histograms based on integral histograms.  

There is also a MATLAB [implementation](https://github.com/yuanj07/FSEG). The results from two implementations are similar. Local spectral histogram computation is coded using pure matrix operations, and thus achieves a speed comparable to the mex c code in MATLAB implementation.  

## Prerequisites

Python 2.7

Numpy

Scipy

Scikit-image

## Usage

To try the code, run 

```sh
python FctSeg.py
```
This verison implements the complete algorithm, which segments an image in a fully automatic fashion. 

To try the version with given seeds, run

```sh
python FctSeg_seed.py
```

Each seed is a pixel location inside one type of texture. Note that this version represents the basic form of the algorithm and does not include nonnegativity constraint. 

Three test images are provided. 
