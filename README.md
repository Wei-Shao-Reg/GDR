# Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts

<img src="images/pipeline.png" align="center" width="800">

This is a C++ implementation of a Geodesic Density Regression (GDR) algorithm presented in the following research paper:

Shao, W., Pan, Y., Durumeric, O.C., Reinhardt, J. M, Bayouth, J. E, Rusu, M., and Christensen, G.E. . "Geodesic Density Regression for Correcting 4DCT Pulmonary Respiratory Motion Artifacts" [[Medical Image Analysis (MedIA)](https://doi.org/10.1016/j.media.2021.102140)]

### Introduction
The GDR algorithm removes respiratory motion artifacts in 4DCT lung images by (1) using binary artifact masks to exclude artifact regions from the regression, and (2) accommodating image intensity change associated with breathing by using a tissue density deformation action.

We have demonstrated that the GDR algorithm is more effective than the state-of-the-art Geodesic Intensity Regression (GIR) algorithm for removing clinically observed motion artifacts in treatment planning 4DCT scans. The following representative result demonstrates the promising performance of GDR in reducing real motion artifact.

<img src="images/GDR_result.PNG" align="center" width="500">

### Computer Requirement
To efficiently run the GDR code, we recommend a machine with a minimum computer memory of 128GB and a minimum number of CPU cores of 16. Our code has been succefully test on MacOS, Linux Mint, and the [[Argon Linux HPC at the University of Iowa] (https://hpc.uiowa.edu/event/63831).]

### Steps to Install the GDR Program
1. Intall proper CMake version (3.1.0 and above) if do not have one ready on your machine https://cmake.org/install/.

2. Install the Insight Toolkit (ITK)
```
1) Download ITK 4.12.2 source code: 
wget https://sourceforge.net/projects/itk/files/itk/4.12/InsightToolkit-4.12.2.tar.gz
tar xvzf InsightToolkit-4.12.2.tar.gz
2) Install ITK
mkdir ITK-build
cd ITK-build
ccmake ../InsightToolkit-4.12.2 by doing c, t, c, g, e, 

make a note for Argon users qlogin

with the following options: [CMAKE_CXX_FLAGS]=-std=c++11, [ITKV3COMPATIBILITY]=ON, [Module_ITKReview]=ON, [Module_ITKV3COMPATIBILITY]=ON.
make -j {number of cores you want to use}
cd ..
```

3. Clone the GDR git repository:
```
git clone https://github.com/Wei-Shao-Reg/GDR.git
mkdir GDR-build
cd GDR-build
ccmake ../GDR/code/ [ITK_DIR] {absolute path of your ITK build}
make
```



### Test Your Code with Simulated 2D CT time series

The "data" folder contains a simulated 2D CT time-series that consists of the 0EX, 20IN, 40IN, 60IN, 80IN, 100IN phases. We have given you the CT images, lung masks, artifact masks, and a parameter file named "param2D.text". A duplication artifat was introduced in the 40IN CT. To remove this duplication artifact, you can run the 2D GDR regression by executing: 

```
mkdir ../GDR/output/
./GDR2D ../GDR/data/param2D.txt
```

The output of GDR regression in "../GDR/output/" includes "artifact-free" CT images, Jacobian images, displacement fields at different breathing phases. By comparing "../GDR/output/template_image_at_time_10.nii.gz" with "../GDR/data/40IN.nii.gz", you can see how the duplication artifact has been removed by the GDR regression.


### How to Run the GDR Program
```
1. Create a parameter file, see parameter/param.txt for an example and parameter/param_exp.txt for an explanation of the parameters.
2. Run 2D GDR regression by executing: GDR2D {your parameter file}
3. Or run 3D GDR regression by executing: GDR3D {your parameter file}
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
