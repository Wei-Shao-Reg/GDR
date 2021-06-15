# Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts
![](images/pipeline.png)

This is a C++ implementation of a Geodesic Density Regression (GDR) algorithm presented in the following research paper:

Shao, W., Pan, Y., Durumeric, O.C., Reinhardt, J. M, Bayouth, J. E, Rusu, M., and Christensen, G.E. . "Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts" [[Accepted to Medical Image Analysis (MedIA)](https://arxiv.org/abs/2106.06853)]

### Introduction
The GDR algorithm removes respiratory motion artifacts in 4DCT lung images by (1) using binary artifact masks to exclude artifact regions from the regression, and (2) accommodating image intesnity change associated with breathing by using a tissue density deformation action.

We have demonstrated that the GDR algorithm is more effective than the state-of-the-art Geodesic Intensity Regression (GIR) algorithm for removing clinically observed motion artifacts in treatment planning 4DCT scans. The following representative result demonstrates the promising performance of GDR in reducing real motion artifact.
![](images/GDR_result.PNG)

### Computater Requirement
To efficiently run the GDR code, we recommend a Linux machine with a minimum computer memory of 256GB and a minimum number of CPU cores of 16.

### Steps to Install the GDR Program
1. Intall proper CMake version (3.1.0 and above) if do not have one ready on your machine https://cmake.org/install/.

2. Install the Insight Toolkit (ITK)
```
1) Download ITK 4.8.2 source code: 
wget https://sourceforge.net/projects/itk/files/itk/4.13/InsightToolkit-4.13.2.tar.gz
tar xvzf InsightToolkit-4.13.2.tar.gz
2) Install ITK
mkdir ITK-bld
cd ITK-bld
ccmake ../InsightToolkit-4.13.2 with the following options: [Module_ITKReview] ON, [Module_ITKV3COMPATIBILITY] ON,and [ITKV3COMPATIBILITY] ON
make
```

3. Clone the GDR git repository:
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
2. Run the GDR regression by executing: GDR3D param.txt
```

### Contact Informaiton
If you have any trouble installing or running the GDR code, you can contact Wei Shao via weishao@stanford.edu.

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
