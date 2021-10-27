# Deep SORT v2

Derived from 

https://github.com/nwojke/deep_sort.git

## Introduction

This repository is an update to tensorflow v2 for the famous deep_sort project.  It is
necessary to convert the model given in the original repository to a v2 format for
tensorflow.  This is accomplished using the convert.py script included.  You will need
to get the original format model from the nwojke/deep_sort repository to test the 
script.  The final v2 saved_model format is included with this repository.

The test program is hard coded for a <a href="https://motchallenge.net/data/MOT16/">MOT 16 Benchmark</a>
sequence. We assume resources have been extracted to the repository root directory and the MOT16 
benchmark data is in ./MOT16:

The v2model.py script will run the deep_sort test on the MOT16 files.  The script has
been streamlined to reduce the code base, but the underlying functionality should be
identical to the original deep_sort program.

Model development and training code has been omitted from this repository, it is only
concerned with run time functionality.

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
