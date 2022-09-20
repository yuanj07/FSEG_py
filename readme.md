# Factorization-based segmentation Python implementation

This is a Python implementation of the factorization-based segmentation algorithm, which fast segments textured images.
The algorithm is described in

**J. Yuan, D. L. Wang, and A. M. Cheriyadat. Factorization-based texture segmentation. IEEE Transactions on Image
Processing, 2015.**

[Here](https://sites.google.com/site/factorizationsegmentation/) is a brief introduction of the
algorithm. [Here](https://medium.com/@jiangye07/fast-local-histogram-computation-using-numpy-array-operations-d96eda02d3c)
is an explanation of computing local histograms based on integral histograms.

There is also a MATLAB [implementation](https://github.com/yuanj07/FSEG). The results from two implementations are
similar. Local spectral histogram computation is coded using pure matrix operations, and thus achieves a speed
comparable to the mex c code in MATLAB implementation.

**OBS** : This repository is a fork for the [original code](https://github.com/yuanj07/FSEG_py)

## TODO

In this actual stage, this code is only reading gray-scale images. Need to move later to read BGR/RGB images

Also, it's only working for .png and .jpg

## Prerequisites

Python 3.9 or above

Numpy>=1.23.3

Scipy>=1.9.1

Scikit-image>=0.19.3

opencv-pytho>=4.6.0.66

## Usage

### Unsupervised texture segmentation mode

This verison implements the complete algorithm, which segments an image in a fully automatic fashion.

To try the code, run

```sh
python3 FctSeg.py -f "img_path" -ws int_value -segn int_value -omega float_value -nonneg_constraint bool_value -save_dir "dir_to_save" -save_file_name "file_name_to_save"
```

where :

**-f** is the image path

**-ws** is the window size value

**-segn** is the number of segment. if set to 0, the number will be automatically estimated

**-omega** is the error threshod for estimating segment number. need to adjust for different filter bank.

**-nonneg_constraint** is a flag to apply the negative matrix factorization

**-save_dir** is the path with the folder to save the file

**-save_file_name** is the file name with the extension to save the final result

#### Example on how to run this code with some parameters

```sh
python3 FctSeg.py -f "Images/img_test.png" -ws 25 -segn 0 -omega 0.045 -nonneg_constraint True -save_dir "ResultImages/" -save_file_name "final_result.png"
```

To try the version with given seeds, run

```sh
python FctSeg_seed.py
```

Each seed is a pixel location inside one type of texture. Note that this version represents the basic form of the
algorithm and does not include nonnegativity constraint.

Three test images are provided. 
