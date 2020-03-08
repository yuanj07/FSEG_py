# Factorization-based segmentation Python implementation

This is a Python implementation of the factorization-based segmentation algorithm, which is described in 

**J. Yuan, D. L. Wang, and A. M. Cheriyadat. Factorization-based texture segmentation. IEEE Transactions on Image Processing, 2015.**

The MATLAB implementation can be found at [here](https://github.com/yuanj07/FSEG). The results from two implementations are similar. Local spectral histogram computation is implemented using pure matrix operations, and thus achieves a speed comparable to the mex c code in MATLAB implementation.  

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

or try the version with given seeds

```sh
python FctSeg_seed.py
```

Each seed is a pixel location inside one type of texture. Note that this version represents the basic form of the algorithm and does not include nonnegativity constraint. 

Three test images are provided. 
