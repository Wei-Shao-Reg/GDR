# Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts

<img src="images/pipeline.png" align="center" width="800">

This is a C++ implementation of the Geodesic Density Regression (GDR) algorithm for 4DCT motion artifacts correction in the following paper:

Shao, W., Pan, Y., Durumeric, O.C., Reinhardt, J. M, Bayouth, J. E, Rusu, M., and Christensen, G.E. . "Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts" [[Medical Image Analysis (MedIA)](https://doi.org/10.1016/j.media.2021.102140)], 2021.

### Introduction
The GDR algorithm removes respiratory motion artifacts in 4DCT lung images by (1) using binary artifact masks to exclude artifact regions from the regression, and (2) accommodating image intensity change associated with breathing by using a tissue density deformation action.

We have demonstrated that the GDR algorithm is more effective than the state-of-the-art Geodesic Intensity Regression (GIR) algorithm for removing clinically observed motion artifacts in treatment planning 4DCT scans. The following representative result demonstrates the promising performance of GDR in reducing real motion artifact.

<img src="images/GDR_result.png" align="center" width="300">

### Computer Requirement
To efficiently run the GDR code, we recommend a machine with a minimum computer memory of 128GB and a minimum number of CPU cores of 16. Our code has been succefully test on MacOS, Linux Mint, and the [[Argon Linux HPC at the University of Iowa](https://hpc.uiowa.edu/event/63831).]

### Steps to Install the GDR Program
1. Install proper CMake (3.1.0 and above) if do not have one on your machine https://cmake.org/install/. Note, if you have already installed Anaconda, you can just run the following command to install CMake:
```
conda install -c anaconda cmake
```

if you are a sudo user in Linux, you can also run the following command to install CMake:
```
sudo apt-get -y install cmake
```

2. Install the Insight Toolkit (ITK)
```
wget https://sourceforge.net/projects/itk/files/itk/4.12/InsightToolkit-4.12.2.tar.gz
tar xvzf InsightToolkit-4.12.2.tar.gz
mkdir ITK-build
cd ITK-build
cmake -DCMAKE_CXX_FLAGS="-std=c++11" -DITKV3_COMPATIBILITY:BOOL=ON -DModule_ITKReview:BOOL=ON -DModule_ITKV3Compatibility:BOOL=ON ../InsightToolkit-4.12.2/
make -j56
cd ..
```

3. Install the GDR algorithm:
```
git clone https://github.com/Wei-Shao-Reg/GDR.git
mkdir GDR-build
cd GDR-build
cmake -DCMAKE_CXX_FLAGS="-std=c++11" -DCMAKE_BUILD_TYPE:STRING="Release" -DITK_DIR=../ITK-build/  ../GDR/code/
make

```

note: set [CMAKE_CXX_FLAGS]=-std=c++11, and set [CMAKE_BUILD_TYPE]=Release will make your code 10 times faster.


### Test Your Code with Simulated 2D CT time series.

The "data" folder contains a simulated 2D CT time-series that consists of the 0EX, 20IN, 40IN, 60IN, 80IN, 100IN phases. We have given you the CT images, lung masks, artifact masks, and a parameter file named "param2D.text". A duplication artifat was introduced in the 40IN CT. To remove this duplication artifact, you can run the 2D GDR regression by executing: 

```
mkdir output
./GDR2D param2D.txt
```

The output of GDR regression in the "output" directory includes "artifact-free" CT images, Jacobian images, displacement fields at different breathing phases. By comparing "../GDR/output/template_image_at_time_10.nii.gz" with "../GDR/data/40IN.nii.gz", you can see how the duplication artifact has been removed by the GDR regression.


### To summarize, use the following two steps to run the GDR algroithm:
```
1. Create a parameter file, see param2D.txt for an example of parameter file for 2D GDR regression, param3D.tx for an example of parameter file for 3D GDR regression, and param_explanation.txt for an explanation of all of the variables.
2. Run 2D GDR regression by executing: ./GDR2D {path to your parameter file} Or run 3D GDR regression by executing: ./GDR3D {path to your parameter file}.
```


### Contact Informaiton
If you have any trouble installing or running the GDR code, you can contact Wei Shao via weishao@stanford.edu.

### BibTeX

If you use this code, please cite the following paper:

```bibtex

@article{Shao_GDR_2021,
title = {Geodesic Density Regression for Correcting {4DCT} Pulmonary Respiratory Motion Artifacts},
journal = {Medical Image Analysis},
pages = {102140},
vol = {72},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102140},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521001869},
author = {Wei Shao and Yue Pan and Oguz C. Durumeric and Joseph M. Reinhardt and John E. Bayouth and Mirabela Rusu and Gary E. Christensen},
keywords = {Geodesic regression, Artifact correction, Motion artifact, 4DCT, Image registration, Lung cancer}
}
```
