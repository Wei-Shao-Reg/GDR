# Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts
![](images/pipeline.png)

This is a C++ implementation of a Geodesic Density Regression (GDR) algorithm presented in the following research paper:

Shao, Wei, et al. "Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts" [Medical Image Analyis]

### Introduction
The GDR algorithm removes respiratory motion artifacts in 4DCT lung images by (1) using binary artifact masks to exclude artifact regions from the regression, and (2) accommodating image intesnity change associated with breathing by using a tissue density deformation action.

This code has been tested on several treatment planing 4DCT scans and the following representative result demonstrates its promising performance in reducing real motion artifact.
![](images/GDR_result.PNG)

### Computation Requirement
To efficiently run the GDR code, we recommend a minimum computer memory of 256GB and a minimum number of CPU cores to be 16.

### How to Install the GDR Program
1. Install the Insight Toolkit (ITK)
```
1) Download ITK 4.8.2 source code: 
wget https://sourceforge.net/projects/itk/files/itk/4.8/InsightToolkit-4.8.2.tar.gz
tar xvzf InsightToolkit-4.8.2.tar.gz
2) Install ITK
mkdir ITK-bld
cd ITK-bld
ccmake ../InsightToolkit-4.8.2 with the following options: [ModuleITKReview] ON, and [ITKV3COMPATIBILITY] ON
```

2. Intall CMake by following instructions in https://cmake.org/install/.

3. Clone the repository:
```
git clone https://github.com/Wei-Shao-Reg/GDR.git
mkdir GDR-bld
cd GDR-bld
ccmake ../GDR/code/
make
```

### How to Run the GDR Program
```
1. Create a parameter file, see parameter/param.txt for an example and parameter/param_exp.txt for an explanation of the parameters.
2. Run the GDR regression by running GDR3D param.txt.
```

### Contact Informaiton
If you have any trouble installing or running the program, you can contact Wei Shao via weishao@stanford.edu.

### BibTeX

If you use this code, please cite the following paper:

```bibtex
@article{Shao_GDR_2021,
	year = 2021,
	author = {Wei Shao and Yue Pan and Oguz C. Durumeric and Joseph M. Reinhardt and John E. Bayouth and Mirabela Rusu and Gary E. Christensen},
	title = {Geodesic Density Regression for Correcting {4DCT} Pulmonary Respiratory Motion Artifacts},
	journal = {Medical Image Analysis}
}
```
