/**
 Geodesic regression of time-series 2D CT images by flow of transformation maps
 @filename GDR2D.cxx
 @author Wei Shao
 @advisor Gary Christensen


If you use this code, please cite the following paper:

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


Licensed under the MIT License:

Copyright 2021 Wei Shao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 */



#include <cstdlib>
//#include <ctime>
#include <random>
//#include <cmath>
#include <iostream>
#include "itkImageFileReader.h"
#include <itkImageFileWriter.h>
#include "itkImage.h"
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include "itkImageRegionIterator.h"
#include <itkVectorIndexSelectionCastImageFilter.h>
#include "itkDerivativeImageFilter.h"
// We need turn on ITKv3 Compatability ON to use
// --- itkMultiplyByConstantImageFilter.h,
// ITK directory: /Users/weishao/src/ITK-bld-4.13.0
#include <itkMultiplyByConstantImageFilter.h>
//#include <itkSubtractImageFilter.h>
#include <itkGradientImageFilter.h>
// sigma is in physical units
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include <itkComposeImageFilter.h>
#include "itkDisplacementFieldTransform.h"
#include "itkResampleImageFilter.h"
//#include "itkStatisticsImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"
//#include <fstream>
//#include "itkDivideImageFilter.h"
#include <chrono>  // for high_resolution_clock

using namespace std;
const unsigned int Dimension = 2; // image dimenstion = 2
////// create aliases for data types and filters
typedef itk::Image<float,Dimension>                                                            ImageType;
typedef itk::ImageFileReader<ImageType>                                                        ImageFileReaderType;
typedef itk::ImageFileWriter< ImageType>                                                       ImageWriterType;
typedef itk::Vector< float, Dimension>                                                         VectorPixelType;
typedef itk::Image< VectorPixelType, Dimension >                                               VectorFieldImageType;
typedef itk::ImageFileWriter< VectorFieldImageType>                                            VectorImageWriterType;
typedef itk::DisplacementFieldTransform<float, Dimension>                                      DisplacementTransformType;
typedef itk::MultiplyImageFilter <ImageType, ImageType >                                       MultiplyImageType;
//typedef itk::MultiplyImageFilter <ImageType, VectorFieldImageType, VectorFieldImageType >      MultiplyVectorImageType;
//typedef itk::MultiplyImageFilter <ImageType, VectorFieldImageType, VectorFieldImageType >      MultiplyScalarAndVectorImageType;
typedef itk::VectorIndexSelectionCastImageFilter<VectorFieldImageType, ImageType>              IndexSelectionType;
typedef itk::DerivativeImageFilter <ImageType, ImageType>                                      DerivativeFilterType;
typedef itk::AddImageFilter< ImageType, ImageType>                                             AddImageType;
typedef itk::MultiplyByConstantImageFilter<ImageType, float, ImageType>                        MultiplyByConstantType;
typedef itk::SubtractImageFilter< ImageType, ImageType >                                       SubtractImageType;
typedef itk::SubtractImageFilter< VectorFieldImageType, VectorFieldImageType >                 SubtractVectorImageType;
typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>                       GaussianFilterType;
//typedef itk::SmoothingRecursiveGaussianImageFilter<VectorFieldImageType, VectorFieldImageType> VectorGaussianFilterType;
typedef itk::AddImageFilter< VectorFieldImageType, VectorFieldImageType>                       AddVectorImageType;
typedef itk::ComposeImageFilter<ImageType, VectorFieldImageType>                               ImageToVectorImageFilterType;
typedef itk::MultiplyByConstantImageFilter<VectorFieldImageType, float, VectorFieldImageType>  MultiplyVectorImageByConstantType;
//typedef itk::GradientImageFilter<ImageType,float>                                              GradientImageType;
typedef itk::ResampleImageFilter<ImageType, ImageType, float>                                  ResampleImageFilterType;
typedef itk::ResampleImageFilter<VectorFieldImageType, VectorFieldImageType>                   ResampleVectorImageFilterType;
//typedef itk::StatisticsImageFilter<ImageType>                                                  StatisticsImageFilterType;
typedef itk::DisplacementFieldJacobianDeterminantFilter<VectorFieldImageType,float,ImageType>  JacobianFilterType;
//typedef itk::DivideImageFilter <ImageType, ImageType, ImageType >                              DivideImageFilterType;

// the function DivideImage is used to compute the ratio of two tissue density images
// the images are in [0,100] float, if the denominator is 1% or below, we make the ratio zero
// the purpose for this is to improve the robustness of our algorithm at the boundaries of the lungs
ImageType::Pointer DivideImage(const ImageType::Pointer& N, const ImageType::Pointer& D){
    itk::ImageRegionIterator<ImageType> imageIterator1(N,N->GetBufferedRegion());
    itk::ImageRegionIterator<ImageType> imageIterator2(D,D->GetBufferedRegion());
    imageIterator1.GoToBegin();
    imageIterator2.GoToBegin();
    while(!imageIterator1.IsAtEnd())
    {
        // Get the current pixel
        float n = imageIterator1.Get();
        float d = imageIterator2.Get();
        float ratio = 0;
        // when the denominator
        if(abs(d)>1){
            ratio = n/d;
        }
        imageIterator1.Set(ratio);
        ++imageIterator1;
        ++imageIterator2;
    }
    return N;
}

float VelocityProduct(std::vector<VectorFieldImageType::Pointer> L, std::vector<VectorFieldImageType::Pointer> R)
{
    float sum = 0;
    for(int i =0; i < L.size(); i++)
    {
        itk::ImageRegionIterator<VectorFieldImageType> imageIterator1(L[i],L[i]->GetBufferedRegion());
        itk::ImageRegionIterator<VectorFieldImageType> imageIterator2(R[i],R[i]->GetBufferedRegion());
        imageIterator1.GoToBegin();
        imageIterator2.GoToBegin();
        while(!imageIterator1.IsAtEnd())
        {
            // Get the current pixel
          //  float l0 = imageIterator1.Get()[0];
          //  float l1 = imageIterator1.Get()[1];
          //  float d0 = imageIterator2.Get()[0];
         //   float d1 = imageIterator2.Get()[1];
            //VectorPixelType a = imageIterator1.Get();
           // VectorPixelType b = imageIterator2.Get();

          //  sum = sum + l0*d0 + l1*d1;
            sum = sum + imageIterator1.Get()*imageIterator2.Get();
            
            ++imageIterator1;
            ++imageIterator2;
        }
        
    }
    //std::cout << "sum:" << sum << std::endl;
    return sum;
}

std::vector<VectorFieldImageType::Pointer>  VelMultiConst(float a, std::vector<VectorFieldImageType::Pointer> L)
{
    for(auto & i : L)
    {
        MultiplyVectorImageByConstantType::Pointer   multiplyByConstant  =  MultiplyVectorImageByConstantType::New();
        multiplyByConstant->SetConstant(a);
        multiplyByConstant->SetInput(i);
        multiplyByConstant->Update();
        i = multiplyByConstant->GetOutput();
    }
    return L;
}



std::vector<VectorFieldImageType::Pointer>  VelocitySubtraction(std::vector<VectorFieldImageType::Pointer> L, std::vector<VectorFieldImageType::Pointer> R)
{
    for(int i =0; i < L.size(); i++)
    {
        SubtractVectorImageType::Pointer      subtractVectorField = SubtractVectorImageType::New();
        subtractVectorField->SetInput1(L[i]);
        subtractVectorField->SetInput2(R[i]);
        subtractVectorField->Update();
        L[i] = subtractVectorField-> GetOutput();
    }
    return L;
}

int main(int argc, char *argv[]) {
    if( argc !=2 )
    {
        std::cerr << "Usage: "<< std::endl;
        std::cerr << argv[0] << " [parameter file]";
        std::cerr << std::endl;
        return EXIT_FAILURE;
    }
    
    // define c1 and c2 used in wolfe condition
    float c1 = 0.0001;
    float c2 = 0.9;
    
    
    std::vector<string> inputImagePaths; // paths of input CT images
    std::vector<string> inputMaskPaths; // paths of the masks for input CT images
    std::vector<string> inputArtifactMaskPaths; // paths of artifact masks for input CT images
    
    std::vector<int> outputDispTimes; // time points at which to output a displacement field
    std::vector<int> numberOfIterations; // a list of the number of iterations at each resolution
    int numberOfResolutions; // number of resolutions used in image registration
    int outputDispStartTime; // time point at which the output displacement fields start
    int numberOfInputs = 0; // variable used to keep track of number of input CT images
    
    string outputDirectory; // path of output directory
    string fixedTemplate; // whether or not fixing the template image during regression
    string outputDispType; // type of output displacment field, Lagrangian for push-forward, Eulerian for pull-back
    std::vector<int> inputImageTimes; // time points for the input CT images
    std::vector<float> imageWeights; // sigma for calculating the weight of the similarity cost at each resolution = 1/sigma^2
    std::vector<float> epsilons; // regression stop criterion at each resolution
    std::vector<int> downsampleFactors; // image downsample factor at each resolution
    std::vector<float> deformationKernelSizes; // deformation kernel size (std of Gaussian) used in each resolution
    std::vector<float>   stepSizes; // step size for gradient descent at each resolution
    std::vector<float>   imageSmoothingFactors; // std of Gaussian for smoothing out input images at each resolution
    string similarityCostType; // SSD or SSTVD?
    
    
    /// begin read parameters from the specified parameter file
    
    std::ifstream File (argv[1]);
    if (File.is_open())
    {
        std::string line;
        while(getline(File, line)){
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find('=');
            string name = line.substr(0, delimiterPos);
            string value = line.substr(delimiterPos + 1);
            if (name == "outputDirectory")
            {
                outputDirectory = value; // get the path of the output directory first
            }
        }
    }
    else {
        std::cerr << "Couldn't open config file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    
    ofstream file;
    file.open (outputDirectory + "results.txt"); // output results to the file outputDirectory/results.txt
    
    file << "read in parameters................\n\n";
    cout << "read in parameters................\n" << endl;
    
    std::ifstream cFile (argv[1]);
    if (cFile.is_open())
    {
        std::string line;
        while(getline(cFile, line)){
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find('=');
            string name = line.substr(0, delimiterPos);
            string value = line.substr(delimiterPos + 1);
            
            std::cout << name << "=" << value << "\n\n";
            
            if (name == "InputImage")
            {
                auto Pos1 = value.find(',');
                
                inputImagePaths.push_back(value.substr(0, Pos1));
                string temp = value.substr(Pos1 + 1);
                auto Pos2 = temp.find('=');
                inputImageTimes.push_back(std::stoi(temp.substr(Pos2+1)));
                numberOfInputs++;
                file << "InputImage = " + value + "\n\n";
            }
            
            if (name == "InputMask")
            {
                auto Pos1 = value.find(',');
                inputMaskPaths.push_back(value.substr(0, Pos1));
                file << "InputMask = " + value + "\n\n";
            }
            
            if (name == "ArtifactMask")
            {
                auto Pos1 = value.find(',');
                inputArtifactMaskPaths.push_back(value.substr(0, Pos1));
                file << "ArtifactMask = " + value + "\n\n";
            }
            
            if (name == "outputDispTimes")
            {
                std::stringstream ss(value);
                file << "outputDispTimes = " + value + "\n\n";
                int i;
                while (ss >> i)
                {
                    outputDispTimes.push_back(i);
                    if (ss.peek() == ',')
                    ss.ignore();
                }
            }
            
            if (name == "similarityCostType")
            {
                similarityCostType = value;
                file << "similarityCostType = " + value + "\n\n";
            }
            
            if (name == "downsampleFactors")
            {
                std::stringstream ss(value);
                file << "image downsample factors = " + value + "\n\n";
                int i;
                while (ss >> i)
                {
                    downsampleFactors.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
            
            if (name == "imageSmoothingFactors")
            {
                std::stringstream ss(value);
                file << "image smoothing factors = " + value + "\n\n";
                int i;
                while (ss >> i)
                {
                    imageSmoothingFactors.emplace_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                // ss.at(i);
            }
            
            if (name == "deformationKernelSizes")
            {
                std::stringstream ss(value);
                file << "Gaussian kernel sizes = " + value + "\n\n";
                float i;
                while (ss >> i)
                {
                    deformationKernelSizes.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
            
            if (name == "imageWeights")
            {
                std::stringstream ss(value);
                file << "image weights = " + value + "\n\n";
                float i;
                while (ss >> i)
                {
                    imageWeights.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
            
            if (name == "stepSizes")
            {
                std::stringstream ss(value);
                file << "step sizes = " + value + "\n\n";
                float i;
                while (ss >> i)
                {
                    stepSizes.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
            
            if (name == "epsilons")
            {
                std::stringstream ss(value);
                file << "epsilons = " + value + "\n\n";
                float i;
                while (ss >> i)
                {
                    epsilons.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
            
            if (name == "numberOfResolutions")
            {
                numberOfResolutions = std::stoi(value);
                file << "number of resolutions = " + value + "\n\n";
            }
            
            if (name == "outputDispStartTime")
            {
                outputDispStartTime = std::stoi(value);
                file << "output displacement field start time = " + value + "\n\n";
            }
            
            if (name == "numberOfIterations")
            {
                std::stringstream ss(value);
                file << "numberOfIterations = " + value + "\n\n";
                float i;
                while (ss >> i)
                {
                    numberOfIterations.emplace_back(i);
                    if (ss.peek() == ',')
                    ss.ignore();
                }
            }
            
            if (name == "fixedTemplate")
            {
                file << "fix template image? = " + value + "\n\n";
                fixedTemplate = value;
            }
            
            if (name == "outputDispType")
            {
                outputDispType = value;
                file << "outputDispType = " + value + "\n\n";
            }
        }
    }
    else {
        std::cerr << "Couldn't open config file \n\n";
        return EXIT_FAILURE;
    }
    /// end read parameters from the specified parameter file

    
    // compute time interval lengths, i.e., step sizes used to solve O.D.Es
    std::vector<float> intervalLengths;
    
    for(int l=0; l< numberOfInputs - 1; l++)
    {
        intervalLengths.emplace_back(1.0/((numberOfInputs-1)*(inputImageTimes.at(l+1) - inputImageTimes.at(l))));
    }
    
    // number of time points used to parameterize transformations
    int numberOfTimeIntervals = inputImageTimes.at(numberOfInputs-1);
    
    const float HU_air = -1000.f; // Hounsfield unit of air
    const float HU_tissue = 55.f; // Hounsfield unit of air tissue
    
    std::vector<ImageType::Pointer>      inputImages; // this is the masked version of input images
    std::vector<ImageType::Pointer>      originalInputImages; // this is the no-masked version of the input CT images
    std::vector<ImageType::Pointer>      inputMasks; // input CT image masks
    std::vector<ImageType::Pointer>      inputArtifactMasks; // input CT artifact masks
    
    //////// begin read in input CT images,masks, and artifact masks
    for (int i=0; i<numberOfInputs;i++)
    {
        ImageFileReaderType:: Pointer   ImageReader = ImageFileReaderType::New();
        ImageReader->SetFileName(inputImagePaths.at(i));
        ImageReader->Update();
        inputImages.emplace_back(ImageReader->GetOutput());
        
        ImageFileReaderType:: Pointer   MaskReader = ImageFileReaderType::New();
        MaskReader->SetFileName(inputMaskPaths.at(i));
        MaskReader->Update();
        inputMasks.emplace_back(MaskReader->GetOutput());
        
        ImageFileReaderType:: Pointer   ArtifactMaskReader = ImageFileReaderType::New();
        ArtifactMaskReader->SetFileName(inputArtifactMaskPaths.at(i));
        ArtifactMaskReader->Update();
        inputArtifactMasks.emplace_back(ArtifactMaskReader->GetOutput());
    }
    //////// end read in input CT images,masks, and artifact masks
    
    //// keep original CT images for computation of final template image
    for (int i=0; i<numberOfInputs;i++)
    {
        ImageFileReaderType:: Pointer   ImageReader = ImageFileReaderType::New();
        ImageReader->SetFileName(inputImagePaths.at(i));
        ImageReader->Update();
        originalInputImages.emplace_back(ImageReader->GetOutput());
    }
    
    ////////
    
    //// get image information
    ImageType::RegionType imageRegion = inputImages[0]->GetBufferedRegion();
    ImageType::SizeType   imageSize =  imageRegion.GetSize();
    ImageType::SpacingType  imageSpacing = inputImages[0]->GetSpacing();
    
    //// convert CT images into tissue density images = (HU - HU_air)/(HU_tissue - HU_air)
    for (int i=0; i<numberOfInputs;i++)
    {
        itk::ImageRegionIterator<ImageType> it(inputImages[i],inputImages[i]->GetBufferedRegion());
        while(!it.IsAtEnd())
        {
            // Set the current pixel
            it.Set( 100*(it.Get() - HU_air ) / ( HU_tissue - HU_air )); // in percentage of tissue (0 - 100)%
            ++it;
        }
    }
    
    //// convert original CT images into tissue density images = (HU - HU_air)/(HU_tissue - HU_air)
    for (int i=0; i<numberOfInputs;i++)
    {
        itk::ImageRegionIterator<ImageType> it(originalInputImages[i],originalInputImages[i]->GetBufferedRegion());
        while(!it.IsAtEnd())
        {
            // Set the current pixel
            it.Set( 100*(it.Get() - HU_air ) / ( HU_tissue - HU_air )); // in percentage of tissue (0 - 100)%
            ++it;
        }
    }
    
    //// Muliply tissue density images by image masks for GDR regression process
    for (int i=0; i<numberOfInputs;i++)
    {
        MultiplyImageType::Pointer         multiplyByMask =  MultiplyImageType::New();
        multiplyByMask->SetInput1(inputImages[i]);
        multiplyByMask->SetInput2(inputMasks[i]);
        multiplyByMask->Update();
        inputImages[i] = multiplyByMask->GetOutput();
    }
    
    
    ///..... begin initialize velocity field for the first resolution
    int initialsampleFactor = downsampleFactors.at(0);
    ImageType::RegionType   imageRegionInitial;
    ImageType::SizeType     imageSizeInitial    = imageSize;
    ImageType::SpacingType  imageSpacingInitial = imageSpacing;
    
    //// downsample the image for the first resolution
    imageSizeInitial[0] = imageSizeInitial[0]/initialsampleFactor;
    imageSizeInitial[1] = imageSizeInitial[1]/initialsampleFactor;
	
    //// inreasing the image spacing for the first resolution
    imageSpacingInitial[0] = imageSpacingInitial[0]*initialsampleFactor;
    imageSpacingInitial[1] = imageSpacingInitial[1]*initialsampleFactor;
    imageRegionInitial.SetSize(imageSizeInitial);
    
    
    std::vector<VectorFieldImageType::Pointer>      velocityField; // time-varying velocity field
    for(int i =0; i <= numberOfTimeIntervals; i++)
    {
        VectorFieldImageType::Pointer  velocityImage = VectorFieldImageType::New();
        velocityField.push_back(velocityImage);
        velocityImage->SetBufferedRegion(imageRegionInitial);
        velocityImage->SetSpacing(imageSpacingInitial);
        velocityImage->SetDirection(inputImages[0]->GetDirection());
        velocityImage->SetOrigin(inputImages[0]->GetOrigin());
        velocityImage->Allocate();
    }
    
    for(int i=0; i<= numberOfTimeIntervals; i++)
    {
        itk::ImageRegionIterator<VectorFieldImageType> displacementFieldImageIterator(velocityField[i], velocityField[i]->GetBufferedRegion());
        while(!displacementFieldImageIterator.IsAtEnd())
        {
            VectorPixelType v;
            v[0] = 0.0;
            v[1] = 0.0;
            displacementFieldImageIterator.Set(v);
            ++displacementFieldImageIterator;
        }
    }
    ///.... end initialize velocity field for the first resolution
    
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    std::cout << "\nstarting image registration\n\n " << std::endl;
    
    file << "\nstarting image registration\n\n ";
    
    int regTimes[numberOfResolutions]; /// keep track of running time at each resolution
    float scale = -1.0;
    float scale_cons = -1.0;
    const unsigned int M = 2;
    
    //// start the optimization procedure
    for (int level = 1; level <= numberOfResolutions; level++)
    {
        scale = scale_cons;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        float myStepSize = -stepSizes.at(level-1);
        float tempStepSize;
        
        float convergeRate = 1.0; // cost function convergence rate
        
        std::cout << "Resolution level: " << level << std::endl;
        
        file << "Resolution level: " + to_string(level) + "\n";
        
        // downsample images at each resolution by a specified factor
        int downsampleFactor = downsampleFactors.at(level-1);
        
        
        ImageType::RegionType imageRegionDown;
        ImageType::SizeType imageSizeDown = imageSize;
        ImageType::SpacingType imageSpacingDown = imageSpacing;
        
        // compute the size and spacing of downsampled image
        imageSizeDown[0] = imageSizeDown[0]/downsampleFactor;
        imageSizeDown[1] = imageSizeDown[1]/downsampleFactor;

        imageSpacingDown[0] = imageSpacingDown[0]*downsampleFactor;
        imageSpacingDown[1] = imageSpacingDown[1]*downsampleFactor;

        
        imageRegionDown.SetSize(imageSizeDown);
        
        std::vector<ImageType::Pointer>      inputImagesDown;
        std::vector<ImageType::Pointer>      inputArtifactMasksDown;
        
        /// initialize update field, used in Armijo linear line search algorithm to update step size, same size as v_t
        /// uses a lot of memory, but it converges faster, more robust
        std::vector<VectorFieldImageType::Pointer>      updateField; // time-varying update velocity field
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            VectorFieldImageType::Pointer  velocityImage = VectorFieldImageType::New();
            updateField.push_back(velocityImage);
            velocityImage->SetBufferedRegion(imageRegionDown);
            velocityImage->SetSpacing(imageSpacingDown);
            velocityImage->SetDirection(inputImages[0]->GetDirection());
            velocityImage->SetOrigin(inputImages[0]->GetOrigin());
            velocityImage->Allocate();
        }
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<VectorFieldImageType> displacementFieldImageIterator(updateField[i], updateField[i]->GetBufferedRegion());
            while(!displacementFieldImageIterator.IsAtEnd())
            {
                VectorPixelType v;
                v[0] = 0.0;
                v[1] = 0.0;
                displacementFieldImageIterator.Set(v);
                ++displacementFieldImageIterator;
            }
        }
        //// finish initialization of update field
        
        
        // ------ added for LBFGS
        
        /// store the past m velocity fields and gradients
        
        
        std::vector<VectorFieldImageType::Pointer>                  Z;
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
            Z.push_back(displacement);
            displacement->SetBufferedRegion(imageRegionDown);
            displacement->SetSpacing(imageSpacingDown);
            displacement->SetDirection(inputImages[0]->GetDirection());
            displacement->SetOrigin(inputImages[0]->GetOrigin());
            displacement->Allocate();
        }
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<VectorFieldImageType> displacementFieldIterator(Z[i], Z[i]->GetBufferedRegion());
            while(!displacementFieldIterator.IsAtEnd())
            {
                VectorPixelType v;
                v[0] = 0.0;
                v[1] = 0.0;
                displacementFieldIterator.Set(v);
                ++displacementFieldIterator;
            }
        }
        
        std::vector<VectorFieldImageType::Pointer>                  q;
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
            q.push_back(displacement);
            displacement->SetBufferedRegion(imageRegionDown);
            displacement->SetSpacing(imageSpacingDown);
            displacement->SetDirection(inputImages[0]->GetDirection());
            displacement->SetOrigin(inputImages[0]->GetOrigin());
            displacement->Allocate();
        }
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<VectorFieldImageType> displacementFieldIterator(q[i], q[i]->GetBufferedRegion());
            while(!displacementFieldIterator.IsAtEnd())
            {
                VectorPixelType v;
                v[0] = 0.0;
                v[1] = 0.0;
                displacementFieldIterator.Set(v);
                ++displacementFieldIterator;
            }
        }
        
        /// initialize gk
        std::vector<std::vector<VectorFieldImageType::Pointer>>     gk;
        for(int j =0; j <= M; j++)
        {
            std::vector<VectorFieldImageType::Pointer>      temp; // time-varying velocity field
            for(int i =0; i <= numberOfTimeIntervals; i++)
            {
                VectorFieldImageType::Pointer  tempImage = VectorFieldImageType::New();
                temp.push_back(tempImage);
                tempImage->SetBufferedRegion(imageRegionDown);
                tempImage->SetSpacing(imageSpacingDown);
                tempImage->SetDirection(inputImages[0]->GetDirection());
                tempImage->SetOrigin(inputImages[0]->GetOrigin());
                tempImage->Allocate();
            }
            gk.push_back(temp);
        }
        for(int j =0; j <= M; j++)
        {
            for(int i=0; i<= numberOfTimeIntervals; i++)
            {
                itk::ImageRegionIterator<VectorFieldImageType> tempImageIterator(gk[j][i], gk[j][i]->GetBufferedRegion());
                while(!tempImageIterator.IsAtEnd())
                {
                    VectorPixelType v;
                    v[0] = 0.0;
                    v[1] = 0.0;
                    tempImageIterator.Set(v);
                    ++tempImageIterator;
                }
            }
        }
        /// initialize xk
        std::vector<std::vector<VectorFieldImageType::Pointer>>     xk;
        for(int j =0; j <= M; j++)
        {
            std::vector<VectorFieldImageType::Pointer>      temp; // time-varying velocity field
            for(int i =0; i <= numberOfTimeIntervals; i++)
            {
                VectorFieldImageType::Pointer  tempImage = VectorFieldImageType::New();
                temp.push_back(tempImage);
                tempImage->SetBufferedRegion(imageRegionDown);
                tempImage->SetSpacing(imageSpacingDown);
                tempImage->SetDirection(inputImages[0]->GetDirection());
                tempImage->SetOrigin(inputImages[0]->GetOrigin());
                tempImage->Allocate();
            }
            xk.push_back(temp);
        }
        for(int j =0; j <= M; j++)
        {
            for(int i=0; i<= numberOfTimeIntervals; i++)
            {
                itk::ImageRegionIterator<VectorFieldImageType> tempImageIterator(xk[j][i], xk[j][i]->GetBufferedRegion());
                while(!tempImageIterator.IsAtEnd())
                {
                    VectorPixelType v;
                    v[0] = 0.0;
                    v[1] = 0.0;
                    tempImageIterator.Set(v);
                    ++tempImageIterator;
                }
            }
        }

        
        // initialize vector of rhok
        std::vector<float> rho(M);
        std::vector<float> alpha(M);


        
        
        // ------ added for LBFGS
        
        
        
        //// downsample artifact masks
        for (int i=0;i<numberOfInputs;i++)
        {
            
            ResampleImageFilterType::Pointer   resampleImage = ResampleImageFilterType::New();
            
            resampleImage->SetSize( imageSizeDown);
            resampleImage->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resampleImage->SetOutputSpacing( imageSpacingDown );
            resampleImage->SetOutputDirection( inputImages[0]->GetDirection() );
            resampleImage->SetDefaultPixelValue(0);
            resampleImage->SetInput(inputArtifactMasks[i]);
            resampleImage->Update();
            inputArtifactMasksDown.emplace_back(resampleImage->GetOutput());
        }
        
        ///// downsample input images
        for (int i=0;i<numberOfInputs;i++)
        {
            /// smooth images before resampling
            if (imageSmoothingFactors.at(level-1)>0)
            {
                GaussianFilterType::Pointer    smoothImage  =  GaussianFilterType::New();
                smoothImage->SetInput(inputImages[i]);
                smoothImage->SetSigma(imageSmoothingFactors.at(level-1)); // sigma is measured in the unites of
                smoothImage->Update();
                
           ///     MultiplyImageType::Pointer         multiplyByMask =  MultiplyImageType::New();
           //     multiplyByMask->SetInput1(smoothImage->GetOutput());
           //     multiplyByMask->SetInput2(inputMasks[i]);
           //     multiplyByMask->Update();
                
                ResampleImageFilterType::Pointer   resampleImage = ResampleImageFilterType::New();
                resampleImage->SetSize( imageSizeDown);
                resampleImage->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                resampleImage->SetOutputSpacing( imageSpacingDown );
                resampleImage->SetOutputDirection( inputImages[0]->GetDirection() );
                resampleImage->SetDefaultPixelValue(0);
           //     resampleImage->SetInput(multiplyByMask->GetOutput());
                resampleImage->SetInput(smoothImage->GetOutput());
                resampleImage->Update();
                inputImagesDown.emplace_back(resampleImage->GetOutput());
            }
            else
            {
                ResampleImageFilterType::Pointer   resampleImage = ResampleImageFilterType::New();
                
                resampleImage->SetSize( imageSizeDown);
                resampleImage->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                resampleImage->SetOutputSpacing( imageSpacingDown );
                resampleImage->SetOutputDirection( inputImages[0]->GetDirection() );
                resampleImage->SetDefaultPixelValue(0);
                resampleImage->SetInput(inputImages[i]);
                resampleImage->Update();
                inputImagesDown.emplace_back(resampleImage->GetOutput());
            }
            
        }
        
        
        //// ....begin initialization of state I, and costate lambda to be all zeros
        std::vector<ImageType::Pointer>                 stateImages;
        
       // stateImages.emplace_back(inputImagesDown[0]);
        
        
        std::vector<ImageType::Pointer>                 costateImages;
        
       // for(int i =1; i <= numberOfTimeIntervals; i++)
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            ImageType::Pointer stateImage = ImageType::New();
            stateImages.emplace_back(stateImage);
            stateImage->SetBufferedRegion(imageRegionDown);
            stateImage->SetSpacing(imageSpacingDown);
            stateImage->SetDirection(inputImages[0]->GetDirection());
            stateImage->SetOrigin(inputImages[0]->GetOrigin());
            stateImage->Allocate();
        }
        
        
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            ImageType::Pointer costateImage = ImageType::New();
            costateImages.emplace_back(costateImage);
            costateImage->SetBufferedRegion(imageRegionDown);
            costateImage->SetSpacing(imageSpacingDown);
            costateImage->SetDirection(inputImages[0]->GetDirection());
            costateImage->SetOrigin(inputImages[0]->GetOrigin());
            costateImage->Allocate();
        }
        
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<ImageType> imageIterator(stateImages[i],stateImages[i]->GetBufferedRegion());
            while(!imageIterator.IsAtEnd())
            {
                // Set the current pixel
                imageIterator.Set(0.0);
                ++imageIterator;
            }
        }
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<ImageType> imageIterator(costateImages[i],costateImages[i]->GetBufferedRegion());
            while(!imageIterator.IsAtEnd())
            {
                // Set the current pixel
                imageIterator.Set(0.0);
                ++imageIterator;
            }
        }
        ///... end initialization of state I, and costate lambda
        
        
        ////... begin initialize forwad (Lagrangian) map flow, phi_t
        std::vector<VectorFieldImageType::Pointer>      dispFieldLagrangian;
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
            dispFieldLagrangian.emplace_back(displacement);
            displacement->SetBufferedRegion(imageRegionDown);
            displacement->SetSpacing(imageSpacingDown);
            displacement->SetDirection(inputImages[0]->GetDirection());
            displacement->SetOrigin(inputImages[0]->GetOrigin());
            displacement->Allocate();
        }
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<VectorFieldImageType> displacementFieldIterator(dispFieldLagrangian[i], dispFieldLagrangian[i]->GetBufferedRegion());
            while(!displacementFieldIterator.IsAtEnd())
            {
                VectorPixelType v;
                v[0] = 0.0;
                v[1] = 0.0;
                displacementFieldIterator.Set(v);
                ++displacementFieldIterator;
            }
        }
        ////... end initialize forwad (Lagrangian) map flow, phi_t
        
        ////... begin initialize inverse(Eulerian) map flow, phi_t^{-1}
        std::vector<VectorFieldImageType::Pointer>      dispFieldEulerian;
        for(int i =0; i <= numberOfTimeIntervals; i++)
        {
            VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
            dispFieldEulerian.emplace_back(displacement);
            displacement->SetBufferedRegion(imageRegionDown);
            displacement->SetSpacing(imageSpacingDown);
            displacement->SetDirection(inputImages[0]->GetDirection());
            displacement->SetOrigin(inputImages[0]->GetOrigin());
            displacement->Allocate();
        }
        
        for(int i=0; i<= numberOfTimeIntervals; i++)
        {
            itk::ImageRegionIterator<VectorFieldImageType> displacementFieldIterator(dispFieldEulerian[i], dispFieldEulerian[i]->GetBufferedRegion());
            while(!displacementFieldIterator.IsAtEnd())
            {
                VectorPixelType v;
                v[0] = 0.0;
                v[1] = 0.0;
                displacementFieldIterator.Set(v);
                ++displacementFieldIterator;
            }
        }
        ////... end initialize inverse(Eulerian) map flow, phi_t^{-1}
        
        int iter = 1; // current iteration number
        
        float cost_pre = 0.0; // cost of previous iteration
        float cost_cur = 0.0; // cost of current iteration
        
        ///// optimization procedure depends on the choice of image similarity cost type
        
        if (similarityCostType=="SSTVD") // if tissue density image is used for regression
        {
           // while (  iter<numberOfIterations.at(level-1) && ( (((cost_pre - cost_cur)/cost_pre)<0 || cost_pre==cost_cur || ((cost_pre - cost_cur)/cost_pre)>epsilons.at(level-1)) || iter<=2) )
            while (  iter<numberOfIterations.at(level-1) && ( convergeRate > epsilons.at(level-1) || iter<=2) && scale/scale_cons>0.01 )
            {
                //cout << "scale/scale_con: " << scale/scale_cons << endl;
                /// solve map flow forward
                for(int l=0; l< numberOfInputs - 1; l++)
                {
                    float intervalLength = intervalLengths[l];
                    for(int i = inputImageTimes.at(l); i< inputImageTimes.at(l+1); i++)
                    {
						// deform each component of v[i] by u[i], u[i+1] = u[i] + v[i](Id + u[i])
						DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
						dispTransform->SetDisplacementField(dispFieldLagrangian[i]);
                    
						// select the x and y compoments of velocityField[i]
						IndexSelectionType::Pointer indexSelectionFilterX = IndexSelectionType::New();
						indexSelectionFilterX->SetIndex(0); // 0 for x direction
						indexSelectionFilterX->SetInput(velocityField[i]);
						indexSelectionFilterX->Update();
                    
						IndexSelectionType::Pointer indexSelectionFilterY = IndexSelectionType::New();
						indexSelectionFilterY->SetIndex(1); // 1 for y direction
						indexSelectionFilterY->SetInput(velocityField[i]);
						indexSelectionFilterY->Update();
                    
						//// deform x,y component of v[i] by u[i]
						ResampleImageFilterType::Pointer   resampleX = ResampleImageFilterType::New();
						resampleX->SetInput(indexSelectionFilterX->GetOutput());
						resampleX->SetSize( imageSizeDown );
						resampleX->SetOutputOrigin(  inputImages[0]->GetOrigin() );
						resampleX->SetOutputSpacing( imageSpacingDown );
						resampleX->SetOutputDirection( inputImages[0]->GetDirection() );
						resampleX->SetTransform(dispTransform);
						resampleX->SetDefaultPixelValue(0);
						resampleX->Update();
                    
						ResampleImageFilterType::Pointer   resampleY = ResampleImageFilterType::New();
						resampleY->SetInput(indexSelectionFilterY->GetOutput());
						resampleY->SetSize( imageSizeDown );
						resampleY->SetOutputOrigin(  inputImages[0]->GetOrigin() );
						resampleY->SetOutputSpacing( imageSpacingDown );
						resampleY->SetOutputDirection( inputImages[0]->GetDirection() );
						resampleY->SetTransform(dispTransform);
						resampleY->SetDefaultPixelValue(0);
						resampleY->Update();
                    
						MultiplyByConstantType::Pointer   multiplyXByIntervalLength  =  MultiplyByConstantType::New();
						multiplyXByIntervalLength->SetConstant(intervalLength);
						multiplyXByIntervalLength->SetInput(resampleX->GetOutput());
						multiplyXByIntervalLength->Update();
                    
						MultiplyByConstantType::Pointer   multiplyYByIntervalLength  =  MultiplyByConstantType::New();
						multiplyYByIntervalLength->SetConstant(intervalLength);
						multiplyYByIntervalLength->SetInput(resampleY->GetOutput());
						multiplyYByIntervalLength->Update();
                    
						ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
						composeImageFilter->SetInput(0,multiplyXByIntervalLength->GetOutput());
						composeImageFilter->SetInput(1,multiplyYByIntervalLength->GetOutput());
						composeImageFilter->Update();
                    
						// u[i+1] = u[i] + v[i](x+u[i])*interval length
						AddVectorImageType::Pointer    addDispFields  = AddVectorImageType::New();
						addDispFields->SetInput1(dispFieldLagrangian[i]);
						addDispFields->SetInput2(composeImageFilter->GetOutput());
						addDispFields->Update();
						dispFieldLagrangian[i+1] = addDispFields->GetOutput();
					}
                }
                //// solve inverse map, we are making assumption that we can compute D\phi[i],this code will fail for large curve flows
                /// this implements Equation (5.15) in Wei Shao's PhD Dissertation
                for(int l=0; l< numberOfInputs - 1; l++)
                {
                    float intervalLength = intervalLengths[l];
                    for(int i = inputImageTimes.at(l); i< inputImageTimes.at(l+1); i++)
                    {
                        // select the x,y compoments of velocityField[i]
                        IndexSelectionType::Pointer VX = IndexSelectionType::New();
                        VX->SetIndex(0); // 0 for x direction
                        VX->SetInput(velocityField[i]);
                        VX->Update();
                        
                        IndexSelectionType::Pointer VY = IndexSelectionType::New();
                        VY->SetIndex(1); // 1 for y direction
                        VY->SetInput(velocityField[i]);
                        VY->Update();
                        
                        // select the x,y,z compoments of phi[i]
                        IndexSelectionType::Pointer UX = IndexSelectionType::New();
                        UX->SetIndex(0); // 0 for x direction
                        UX->SetInput(dispFieldLagrangian[i]);
                        UX->Update();
                        
                        IndexSelectionType::Pointer UY = IndexSelectionType::New();
                        UY->SetIndex(1); // 1 for y direction
                        UY->SetInput(dispFieldLagrangian[i]);
                        UY->Update();
                        
                        //... begin compute Jacobian of phi[i]
                        DerivativeFilterType::Pointer       UXxDerivative = DerivativeFilterType::New();
                        UXxDerivative->SetInput(UX->GetOutput());
                        UXxDerivative->SetDirection(0);
                        UXxDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UXyDerivative = DerivativeFilterType::New();
                        UXyDerivative->SetInput(UX->GetOutput());
                        UXyDerivative->SetDirection(1);
                        UXyDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UYxDerivative = DerivativeFilterType::New();
                        UYxDerivative->SetInput(UY->GetOutput());
                        UYxDerivative->SetDirection(0);
                        UYxDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UYyDerivative = DerivativeFilterType::New();
                        UYyDerivative->SetInput(UY->GetOutput());
                        UYyDerivative->SetDirection(1);
                        UYyDerivative->Update();
                        
                        //... end compute Jacobian of phi[i]
                        
                        
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                        
                        //... begin deform each component of Dphi[i] by phi^{-1}[i]
                        ResampleImageFilterType::Pointer   resample11 = ResampleImageFilterType::New();
                        resample11->SetInput(UXxDerivative->GetOutput());
                        resample11->SetSize( imageSizeDown );
                        resample11->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample11->SetOutputSpacing( imageSpacingDown );
                        resample11->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample11->SetTransform(dispTransform);
                        resample11->SetDefaultPixelValue(0);
                        resample11->Update();
                        ImageType::Pointer D11 = resample11->GetOutput();
                        
                        ResampleImageFilterType::Pointer   resample12 = ResampleImageFilterType::New();
                        resample12->SetInput(UXyDerivative->GetOutput());
                        resample12->SetSize( imageSizeDown );
                        resample12->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample12->SetOutputSpacing( imageSpacingDown );
                        resample12->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample12->SetTransform(dispTransform);
                        resample12->SetDefaultPixelValue(0);
                        resample12->Update();
                        ImageType::Pointer D12 = resample12->GetOutput();
                        

                        ResampleImageFilterType::Pointer   resample21 = ResampleImageFilterType::New();
                        resample21->SetInput(UYxDerivative->GetOutput());
                        resample21->SetSize( imageSizeDown );
                        resample21->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample21->SetOutputSpacing( imageSpacingDown );
                        resample21->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample21->SetTransform(dispTransform);
                        resample21->SetDefaultPixelValue(0);
                        resample21->Update();
                        ImageType::Pointer D21 = resample21->GetOutput();
                        
                        ResampleImageFilterType::Pointer   resample22 = ResampleImageFilterType::New();
                        resample22->SetInput(UYyDerivative->GetOutput());
                        resample22->SetSize( imageSizeDown );
                        resample22->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample22->SetOutputSpacing( imageSpacingDown );
                        resample22->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample22->SetTransform(dispTransform);
                        resample22->SetDefaultPixelValue(0);
                        resample22->Update();
                        ImageType::Pointer D22 = resample22->GetOutput();
                        // .....end deform components of Dphi[i]
                        
                        //... compute the inverse of Jacobian matrix Dphi[i]\circ\phi^{-1}[i]
                        
                        itk::ImageRegionIterator<ImageType> iterator11(D11,D11->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator12(D12,D12->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator21(D21,D21->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator22(D22,D22->GetBufferedRegion());
						
                        iterator11.GoToBegin();
                        iterator12.GoToBegin();
                        iterator21.GoToBegin();
                        iterator22.GoToBegin();

                        while(!iterator11.IsAtEnd())
                        {
                            // Get the current pixel
                            float a = 1+ iterator11.Get();
                            float b = iterator12.Get();
                            float c = iterator21.Get();
                            float d = 1+ iterator22.Get();
                           
                            float det = a*d - b*c;
                            
                            iterator11.Set(d/det);
                            iterator12.Set(-b/det);

                            
                            iterator21.Set(-c/det);
                            iterator22.Set(a/det);

                            ++iterator11;
                            ++iterator12;
                            ++iterator21;
                            ++iterator22;
                        }
                        
                        //.. end
                        
                        /// begin multiply the Jacobian matrix by v[i]
                        MultiplyImageType::Pointer         multiplyXX =  MultiplyImageType::New();
                        multiplyXX->SetInput1(D11);
                        multiplyXX->SetInput2(VX->GetOutput());
                        multiplyXX->Update();
                        
                        MultiplyImageType::Pointer         multiplyXY =  MultiplyImageType::New();
                        multiplyXY->SetInput1(D12);
                        multiplyXY->SetInput2(VY->GetOutput());
                        multiplyXY->Update();
                        
                        
                        AddImageType::Pointer    addX1 = AddImageType::New();
                        addX1->SetInput1(multiplyXX->GetOutput());
                        addX1->SetInput2(multiplyXY->GetOutput());
                        addX1->Update();
                               
                        
                        MultiplyImageType::Pointer         multiplyYX =  MultiplyImageType::New();
                        multiplyYX->SetInput1(D21);
                        multiplyYX->SetInput2(VX->GetOutput());
                        multiplyYX->Update();
                        
                        MultiplyImageType::Pointer         multiplyYY =  MultiplyImageType::New();
                        multiplyYY->SetInput1(D22);
                        multiplyYY->SetInput2(VY->GetOutput());
                        multiplyYY->Update();
                        
                        
                        AddImageType::Pointer    addY1 = AddImageType::New();
                        addY1->SetInput1(multiplyYX->GetOutput());
                        addY1->SetInput2(multiplyYY->GetOutput());
                        addY1->Update();
                        
                        /// end multiply the Jacobian matrix by v[i]
                        
                        /// multiply by -interval length
                        MultiplyByConstantType::Pointer   multiplyXByIntervalLength  =  MultiplyByConstantType::New();
                        multiplyXByIntervalLength->SetConstant(-intervalLength);
                        multiplyXByIntervalLength->SetInput(addX1->GetOutput());
                        multiplyXByIntervalLength->Update();
                        
                        MultiplyByConstantType::Pointer   multiplyYByIntervalLength  =  MultiplyByConstantType::New();
                        multiplyYByIntervalLength->SetConstant(-intervalLength);
                        multiplyYByIntervalLength->SetInput(addY1->GetOutput());
                        multiplyYByIntervalLength->Update();
                        
                        
                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,multiplyXByIntervalLength->GetOutput());
                        composeImageFilter->SetInput(1,multiplyYByIntervalLength->GetOutput());
                        composeImageFilter->Update();
                        
                        // phi^{-1}[i+1] = phi^{-1}[i] + Delta_t * ((-1)Dphi[i]\circ\phi^{-1}[i])^{-1}v[i]
                        AddVectorImageType::Pointer    addDispFields  = AddVectorImageType::New();
                        addDispFields->SetInput1(dispFieldEulerian[i]);
                        addDispFields->SetInput2(composeImageFilter->GetOutput());
                        addDispFields->Update();
                        dispFieldEulerian[i+1] = addDispFields->GetOutput();
                    }
                }
                /// end computation of inverse flow
                
                /// initialize the template image, same size as the downsampled CT
                ImageType::Pointer   templateImage = ImageType::New();
                templateImage->SetBufferedRegion(imageRegionDown);
                templateImage->SetSpacing(imageSpacingDown);
                templateImage->SetDirection(inputImages[0]->GetDirection());
                templateImage->SetOrigin(inputImages[0]->GetOrigin());
                templateImage->Allocate();
                
                /// weight involved in computation of the template
                ImageType::Pointer   weightImage = ImageType::New();
                weightImage->SetBufferedRegion(imageRegionDown);
                weightImage->SetSpacing(imageSpacingDown);
                weightImage->SetDirection(inputImages[0]->GetDirection());
                weightImage->SetOrigin(inputImages[0]->GetOrigin());
                weightImage->Allocate();
                
                itk::ImageRegionIterator<ImageType> templateImageIterator(templateImage,templateImage->GetBufferedRegion());
                while(!templateImageIterator.IsAtEnd())
                {
                    // Set the current pixel
                    templateImageIterator.Set(0.0);
                    ++templateImageIterator;
                }
                
                itk::ImageRegionIterator<ImageType> weightImageIterator(weightImage,weightImage->GetBufferedRegion());
                while(!weightImageIterator.IsAtEnd())
                {
                    // Set the current pixel
                    weightImageIterator.Set(0.0);
                    ++weightImageIterator;
                }
                
                //// if fixedTemplate = false, update template image at every iteration
                if (fixedTemplate=="False")
                {
                    //// deform masked CT images by Lagrangian disp fields, and sum them up
                    /// the follow lines implement Equation (B.18) in Wei's PhD dissertation
                    for (int i=0; i<numberOfInputs; i++)
                    {
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(i)]);
                        
                        MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                        multiplyByArtifactMask->SetInput1(inputImagesDown[i]);
                        multiplyByArtifactMask->SetInput2(inputArtifactMasksDown[i]);
                        multiplyByArtifactMask->Update();
                        
                        
                        ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                        resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                        resampleI->SetSize( imageSizeDown );
                        resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleI->SetOutputSpacing( imageSpacingDown );
                        resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleI->SetTransform(dispTransform);
                        resampleI->SetDefaultPixelValue(0);
                        resampleI->Update();
                        
                        AddImageType::Pointer    addImage = AddImageType::New();
                        addImage->SetInput1(templateImage);
                        addImage->SetInput2(resampleI->GetOutput());
                        addImage->Update();
                        
                        templateImage = addImage->GetOutput();
                    }
                    
                    /// compute weight used in update of template image
                    for (int i=0; i<numberOfInputs; i++)
                    {
                        
                        ////....start estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                        JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                        JacobianImage->SetInput(dispFieldLagrangian[inputImageTimes.at(i)]);
                        JacobianImage->Update();
                        
                        DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
                        defJac->SetDisplacementField(dispFieldEulerian[inputImageTimes.at(i)]);
                        
                        ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
                        resampleJac->SetInput(JacobianImage->GetOutput());
                        resampleJac->SetSize( imageSizeDown );
                        resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleJac->SetOutputSpacing( imageSpacingDown );
                        resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleJac->SetTransform(defJac);
                        resampleJac->SetDefaultPixelValue(1);
                        resampleJac->Update();
                        
                        ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
                        
                        itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
                        imageIterator.GoToBegin();
                        while(!imageIterator.IsAtEnd())
                        {
                            // Get the current pixel
                            float n = imageIterator.Get();
                            imageIterator.Set(1/n);
                            ++imageIterator;
                        }
                        ////....finish estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                        
                        
                        MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                        multiplyByArtifactMask->SetInput1(myJacobian);
                        multiplyByArtifactMask->SetInput2(inputArtifactMasksDown[i]);
                        multiplyByArtifactMask->Update();
                        
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(i)]);
                        
                        ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                        resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                        resampleI->SetSize( imageSizeDown );
                        resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleI->SetOutputSpacing( imageSpacingDown );
                        resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleI->SetTransform(dispTransform);
                        resampleI->SetDefaultPixelValue(0);
                        resampleI->Update();
                        
                        AddImageType::Pointer    addImage = AddImageType::New();
                        addImage->SetInput1(weightImage);
                        addImage->SetInput2(resampleI->GetOutput());
                        addImage->Update();
                        weightImage = addImage->GetOutput();
                    }
                    
                    templateImage = DivideImage(templateImage,weightImage);
                }
                //// if template is fixed, choose I[0] as the template
                else
                {
                    templateImage = inputImagesDown[0];
                }
                
                
                //// compute flow of state images
                /// this implements Equation (5.6)
                for(int i=0; i<= numberOfTimeIntervals; i++)
                {
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                    // compute I_T(\phi_t^-1)
                    ResampleImageFilterType::Pointer   resampleIT = ResampleImageFilterType::New();
                    // resampleI_T->SetInput(inputImagesDown[0]);
                    resampleIT->SetInput(templateImage);
                    resampleIT->SetSize( imageSizeDown );
                    resampleIT->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleIT->SetOutputSpacing( imageSpacingDown );
                    resampleIT->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleIT->SetTransform(dispTransform);
                    resampleIT->SetDefaultPixelValue(0);
                    resampleIT->Update();
                    
                    
                    //// estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                    /// this implements Equation (5.16) in Wei's PhD thesis
                    /// this avoids computing second derivatives of \phi[i]
                    JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                    JacobianImage->SetInput(dispFieldLagrangian[i]);
                    JacobianImage->Update();
                    
                    DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
                    defJac->SetDisplacementField(dispFieldEulerian[i]);
                    
                    ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
                    resampleJac->SetInput(JacobianImage->GetOutput());
                    resampleJac->SetSize( imageSizeDown );
                    resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleJac->SetOutputSpacing( imageSpacingDown );
                    resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleJac->SetTransform(defJac);
                    resampleJac->SetDefaultPixelValue(1);
                    resampleJac->Update();
                    
                    ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
                    
                    
                    itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
                    imageIterator.GoToBegin();
                    while(!imageIterator.IsAtEnd())
                    {
                        // Get the current pixel value
                        float voxelValue = imageIterator.Get();
                        imageIterator.Set(1/voxelValue);
                        ++imageIterator;
                    }

                    
                    MultiplyImageType::Pointer         multiplyByJacobian =  MultiplyImageType::New();
                    multiplyByJacobian->SetInput1(resampleIT->GetOutput());
                    multiplyByJacobian->SetInput2(myJacobian);
                    multiplyByJacobian->Update();
                    stateImages[i] = multiplyByJacobian->GetOutput();
                }
                //// end computation of the state image flow
                
                float SSD = 0.0; // mean of the sum of squared tissue differences
                float count = 0; // number of voxels
                
                float temp;
                
                //// compute image similarity cost, sum up for I0,I1,I2,...,IN-1
                for (int i=0; i<numberOfInputs; i++)
                {
                    SubtractImageType::Pointer    getDiffImage  =  SubtractImageType::New();
                    getDiffImage->SetInput1(stateImages[inputImageTimes.at(i)]);
                    getDiffImage->SetInput2(inputImagesDown[i]);
                    getDiffImage->Update();
                    

                    MultiplyImageType::Pointer         multiplyByArtifactMask1 =  MultiplyImageType::New();
                    multiplyByArtifactMask1->SetInput1(getDiffImage->GetOutput());
                    multiplyByArtifactMask1->SetInput2(inputArtifactMasksDown[i]);
                    multiplyByArtifactMask1->Update();
                    
                    MultiplyImageType::Pointer     squareDiffImage =  MultiplyImageType::New();
                    squareDiffImage->SetInput1(multiplyByArtifactMask1->GetOutput());
                    squareDiffImage->SetInput2(multiplyByArtifactMask1->GetOutput());
                    squareDiffImage->Update();
                    ImageType::Pointer    squareImage = squareDiffImage->GetOutput();
                    
                    itk::ImageRegionIterator<ImageType> imageIterator(squareImage,squareImage->GetBufferedRegion());
                    while(!imageIterator.IsAtEnd())
                    {
                        // Set the current pixel
                        temp = imageIterator.Get();
                        if (temp>0.01) // only compute SSD in the lung
                        {
                            SSD = SSD + temp;
                            count++;
                        }
                        ++imageIterator;
                    }
                    
                }
                
                cost_cur = SSD/count; // compute mean SSD over all input images
                
                /// output loss, stepsize at current iteration
                if (iter<=1)
                {
                    std::cout << "Iteration number=" << iter << ", Similarity Cost=" << SSD/count << ", Reduction rate=" << ", Step size=" << -1.0*myStepSize << std::endl;
                    file << "Iteration number=" + to_string(iter) + ", Similarity Cost=" + to_string(SSD/count) + ", Reduction rate= " + ", step size= " + to_string(-1.0*myStepSize) + "\n";
                }
                else
                {
                    std::cout << "Iteration number=" << iter << ", Similarity Cost=" << SSD/count << ", Reduction rate=" << (cost_pre - cost_cur)/cost_pre << ", Step size=" << -1.0*myStepSize <<std::endl;
                    file << "Iteration number=" + to_string(iter) + ", Similarity Cost=" + to_string(SSD/count) + ", Reduction rate=" + to_string((cost_pre - cost_cur)/cost_pre) + " step size=" + to_string(-1.0*myStepSize) + "\n";
                }
            
                
                // get lambda[t_N] = (2/(sigma^2)(I(1) - I_1))*M_1 (costate)
                // this implements line 5 in Algorithm 4 in Wei's Phd thesis
                SubtractImageType::Pointer    getCostateError  =  SubtractImageType::New();
                getCostateError->SetInput1(stateImages[numberOfTimeIntervals]);
                getCostateError->SetInput2(inputImagesDown[numberOfInputs-1]);
                getCostateError->Update();
                
                MultiplyByConstantType::Pointer  multiplyByWeight =  MultiplyByConstantType::New();
                multiplyByWeight->SetConstant(2.0f/(imageWeights.at(level-1)*imageWeights.at(level-1)));
                multiplyByWeight->SetInput(getCostateError->GetOutput());
                multiplyByWeight->Update();
                costateImages[numberOfTimeIntervals] = multiplyByWeight-> GetOutput();
                
                MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                multiplyByArtifactMask->SetInput1(costateImages[numberOfTimeIntervals]);
                multiplyByArtifactMask->SetInput2(inputArtifactMasksDown[numberOfInputs-1]);
                multiplyByArtifactMask->Update();
                costateImages[numberOfTimeIntervals] = multiplyByArtifactMask->GetOutput();
                
                ///.... begin compute flow of costates, implements line 7-11 in Algorithm 4 in Wei's thesis
                /// start image changes for every interval [t_{k-1}, t_{k}]
                ImageType::Pointer      startImage = costateImages[numberOfTimeIntervals];
                
                
                int k = numberOfInputs-1;
                while (k>0) {
                    
                    /// compute lambda(t) in the interval [t_{k-1}, t_{k}]
                    for(int i = inputImageTimes.at(k-1); i< inputImageTimes.at(k); i++)
                    {
                        DisplacementTransformType::Pointer    dispTransformPhik = DisplacementTransformType::New();
                        dispTransformPhik->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(k)]);
                        
                        ResampleImageFilterType::Pointer   resampleLambda = ResampleImageFilterType::New();
                        resampleLambda->SetInput(startImage);
                        resampleLambda->SetSize( imageSizeDown );
                        resampleLambda->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleLambda->SetOutputSpacing( imageSpacingDown );
                        resampleLambda->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleLambda->SetTransform(dispTransformPhik);
                        resampleLambda->SetDefaultPixelValue(0);
                        resampleLambda->Update();
                        
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                        
                        /// resample by phi_t^{-1}
                        
                        ResampleImageFilterType::Pointer   resampleImage = ResampleImageFilterType::New();
                        resampleImage->SetInput(resampleLambda->GetOutput());
                        resampleImage->SetSize( imageSizeDown );
                        resampleImage->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleImage->SetOutputSpacing( imageSpacingDown );
                        resampleImage->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleImage->SetTransform(dispTransform);
                        resampleImage->SetDefaultPixelValue(0);
                        resampleImage->Update();
                        costateImages[i] = resampleImage->GetOutput();
                    }
                    
                    k=k-1;
                    
                    //// compute lambda(ti-) = lambda(ti-) + 2/(sigma^2)(I(t_i) - Ii)
                    // get lambda[t_N] = 2/(sigma^2)(I(1) - I_1)
                    SubtractImageType::Pointer    costateError  =  SubtractImageType::New();
                    costateError->SetInput1(stateImages[inputImageTimes.at(k)]);
                    costateError->SetInput2(inputImagesDown[k]);
                    costateError->Update();
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask2 =  MultiplyImageType::New();
                    multiplyByArtifactMask2->SetInput1(costateError->GetOutput());
                    multiplyByArtifactMask2->SetInput2(inputArtifactMasksDown[k]);
                    multiplyByArtifactMask2->Update();
                    
                    MultiplyByConstantType::Pointer  weight =  MultiplyByConstantType::New();
                    weight->SetConstant(2.0f / (imageWeights.at(level - 1) * imageWeights.at(level - 1)));
                    weight->SetInput(multiplyByArtifactMask2->GetOutput());
                    weight->Update();
                    
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(weight->GetOutput());
                    addImage->SetInput2(costateImages[inputImageTimes.at(k)]);
                    addImage->Update();
                    
                    startImage = addImage->GetOutput();
                }

                ///... end compute flow of costates
                
                
                
                //// compute cost reduction rate
                if(cost_cur<cost_pre)
                {
                    convergeRate = (cost_pre - cost_cur)/cost_pre;
                }
                
                //// Update Velocity Field by gradient descent
                /// v_t = v_t - s* (v_t + K*(I(t)Grad(lambda(t))))
                /// We will Gaussian filter to each compnengt of I(t)Grad(\lambda(t))
                
                if (iter >= M + 2)
                {
                    //compute the gradient based on current estimation of v_t
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        // std::cout << "update vector field: " << i << std::endl;
                        // update field = v_t + K*(I(t)grad(lambda(t))), see line 12 in Algorithm 4, in Wei's thesis
                        //compute x,y,Z derivatives of costate image
                        DerivativeFilterType::Pointer       xDerivative = DerivativeFilterType::New();
                        DerivativeFilterType::Pointer       yDerivative = DerivativeFilterType::New();
                        
                        xDerivative->SetInput(costateImages[i]);
                        xDerivative->SetDirection(0);
                        xDerivative->Update();
                        yDerivative->SetInput(costateImages[i]);
                        yDerivative->SetDirection(1);
                        yDerivative->Update();
                        
                        MultiplyImageType::Pointer      multiplyXComponent = MultiplyImageType::New();
                        multiplyXComponent->SetInput1(stateImages[i]);
                        multiplyXComponent->SetInput2(xDerivative->GetOutput());
                        multiplyXComponent->Update();
                        ImageType::Pointer     xProd = multiplyXComponent->GetOutput();
                        
                        MultiplyImageType::Pointer      multiplyYComponent = MultiplyImageType::New();
                        multiplyYComponent->SetInput1(stateImages[i]);
                        multiplyYComponent->SetInput2(yDerivative->GetOutput());
                        multiplyYComponent->Update();
                        ImageType::Pointer     yProd = multiplyYComponent->GetOutput();
                        //   float myKernelSize = gaussianKernelSize*pow(2,(numberOfResolutions-level));
                        float myKernelSize = deformationKernelSizes.at(level-1);
                        
                        GaussianFilterType::Pointer    filterXProd  =  GaussianFilterType::New();
                        filterXProd->SetInput(xProd);
                        filterXProd->SetSigma(myKernelSize);
                        filterXProd->Update();
                        
                        GaussianFilterType::Pointer    filterYProd  =  GaussianFilterType::New();
                        filterYProd->SetInput(yProd);
                        filterYProd->SetSigma(myKernelSize);
                        filterYProd->Update();

                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,filterXProd->GetOutput());
                        composeImageFilter->SetInput(1,filterYProd->GetOutput());
                        composeImageFilter->Update();
                        
                        AddVectorImageType::Pointer     addVectorField  =  AddVectorImageType::New();
                        addVectorField->SetInput1(composeImageFilter->GetOutput());
                        addVectorField->SetInput2(velocityField[i]);
                        addVectorField->Update();

                        updateField[i] = addVectorField->GetOutput();
                    }
                    
                    
                    
                    // check if wolfe condition is satified
                    float RHS1 = cost_pre - c1*myStepSize*VelocityProduct(Z,gk[M]);
                    float LHS1 = cost_cur;
                    
                    float RHS2 = std::abs(c2*VelocityProduct(Z, gk[M]));
                    float LHS2 = std::abs(VelocityProduct(Z, updateField));
                    
                    //std::cout << "LHS1: " << LHS1 << std::endl;
                    //std::cout << "RHS1: " << RHS1 << std::endl;
                    //std::cout << "LHS2: " << LHS2 << std::endl;
                    //std::cout << "RHS2: " << RHS2 << std::endl;
                    
                    // if the wolfe condition is satified, continue to the next iteration
                    if (LHS1<= RHS1 && LHS2<=RHS2 && cost_cur < cost_pre)
                    {
                        // reset scale
                        scale = scale_cons;
                        
                        /// shift elements in gk and xk by 1
                        for (int i=0; i<M; i++)
                        {
                            xk[i] = xk[i+1];
                            gk[i] = gk[i+1];
                        }
                        
                        for (int i=0; i<M-1; i++)
                        {
                            rho[i] = rho[i+1];
                        }
                        
                        
                        xk[M] = velocityField;
                        gk[M] = updateField;
                        
                        q = gk[M];
                         float gamma_k;
                         
                         for(int j=M-1; j>=0; j--)
                         {
                             std::vector<VectorFieldImageType::Pointer> y_k = VelocitySubtraction(gk[j+1], gk[j]);
                             std::vector<VectorFieldImageType::Pointer> s_k = VelocitySubtraction(xk[j+1], xk[j]);
                             if (j == M-1)
                             {
                                 rho[j] = 1.0f/VelocityProduct(y_k,s_k);
                             }
                             
                             
                             alpha[j] = rho[j]*VelocityProduct(s_k,q);
                             q = VelocitySubtraction(q, VelMultiConst(alpha[j],y_k));
                         }
                         std::vector<VectorFieldImageType::Pointer> y_k = VelocitySubtraction(gk[M], gk[M-1]);
                         std::vector<VectorFieldImageType::Pointer> s_k = VelocitySubtraction(xk[M], xk[M-1]);
                         gamma_k = VelocityProduct(s_k, y_k)/VelocityProduct(y_k,y_k);
                         Z = VelMultiConst(gamma_k,q);
                         
                         for(int j=0; j<=M-1; j++)
                         {
                             std::vector<VectorFieldImageType::Pointer> velocitySubtraction = VelocitySubtraction(gk[j + 1], gk[j]);
                             std::vector<VectorFieldImageType::Pointer> subtraction = VelocitySubtraction(xk[j + 1], xk[j]);
                             
                             float beta_j = rho[j]*VelocityProduct(velocitySubtraction, Z);
                             
                             Z = VelocitySubtraction(Z, VelMultiConst( -alpha[j] + beta_j , subtraction));
                         }
                         // update velocity field based on Z
                         
                         /// the followng code is gradient descent for updating v[t]
                        for (int i=0; i<=numberOfTimeIntervals; i++)
                        {
                            // multiply by linear search step size
                            MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                            multiplyByStepSize->SetConstant(scale);
                            multiplyByStepSize->SetInput(Z[i]);
                            multiplyByStepSize->Update();
                            // update v[i]
                            AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                            addUpdateField->SetInput1(velocityField[i]);
                            addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                            addUpdateField->Update();
                            velocityField[i] = addUpdateField->GetOutput();
                        }
                        iter++;
                        cost_pre = cost_cur;
                    }
                    else
                    {
                        /// if the cost is not decreasing, we need to choose a smaller step size
                         scale  = scale*0.5f;
                        // std::cout<< "step size too large: "<<  std::endl;
                         for (int i=0; i<=numberOfTimeIntervals; i++)
                         {
                             
                             // multiply by linear search step size
                             MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                             /// re-implement in 3D
                            // std::cout << "scale:" << scale << std::endl;
                             multiplyByStepSize->SetConstant(-scale);
                             multiplyByStepSize->SetInput(Z[i]);
                             multiplyByStepSize->Update();
                             // update v[i]
                             AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                             addUpdateField->SetInput1(velocityField[i]);
                             addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                             addUpdateField->Update();
                             velocityField[i] = addUpdateField->GetOutput();
                        }
                    }
                }
                
                // compute direction Z for the first time,
                if (iter==M + 1)
                {
                    scale = scale_cons;
                
                    
                    //compute the gradient based on current estimation of v_t
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        // std::cout << "update vector field: " << i << std::endl;
                        // update field = v_t + K*(I(t)grad(lambda(t))), see line 12 in Algorithm 4, in Wei's thesis
                        //compute x,y,Z derivatives of costate image
                        DerivativeFilterType::Pointer       xDerivative = DerivativeFilterType::New();
                        DerivativeFilterType::Pointer       yDerivative = DerivativeFilterType::New();
                        
                        xDerivative->SetInput(costateImages[i]);
                        xDerivative->SetDirection(0);
                        xDerivative->Update();
                        yDerivative->SetInput(costateImages[i]);
                        yDerivative->SetDirection(1);
                        yDerivative->Update();
                        
                        MultiplyImageType::Pointer      multiplyXComponent = MultiplyImageType::New();
                        multiplyXComponent->SetInput1(stateImages[i]);
                        multiplyXComponent->SetInput2(xDerivative->GetOutput());
                        multiplyXComponent->Update();
                        ImageType::Pointer     xProd = multiplyXComponent->GetOutput();
                        
                        MultiplyImageType::Pointer      multiplyYComponent = MultiplyImageType::New();
                        multiplyYComponent->SetInput1(stateImages[i]);
                        multiplyYComponent->SetInput2(yDerivative->GetOutput());
                        multiplyYComponent->Update();
                        ImageType::Pointer     yProd = multiplyYComponent->GetOutput();
                        //   float myKernelSize = gaussianKernelSize*pow(2,(numberOfResolutions-level));
                        float myKernelSize = deformationKernelSizes.at(level-1);
                        
                        GaussianFilterType::Pointer    filterXProd  =  GaussianFilterType::New();
                        filterXProd->SetInput(xProd);
                        filterXProd->SetSigma(myKernelSize);
                        filterXProd->Update();
                        
                        GaussianFilterType::Pointer    filterYProd  =  GaussianFilterType::New();
                        filterYProd->SetInput(yProd);
                        filterYProd->SetSigma(myKernelSize);
                        filterYProd->Update();

                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,filterXProd->GetOutput());
                        composeImageFilter->SetInput(1,filterYProd->GetOutput());
                        composeImageFilter->Update();
                        
                        AddVectorImageType::Pointer     addVectorField  =  AddVectorImageType::New();
                        addVectorField->SetInput1(composeImageFilter->GetOutput());
                        addVectorField->SetInput2(velocityField[i]);
                        addVectorField->Update();

                        updateField[i] = addVectorField->GetOutput();
                    }
                    
                    xk[M] = velocityField;
                    gk[M] = updateField;
                    
                    
                    q = gk[M];
                    float gamma_k;
                    
                    
                    for(int j=M-1; j>=0; j--)
                    {
                        std::vector<VectorFieldImageType::Pointer> y_k = VelocitySubtraction(gk[j+1], gk[j]);
                        std::vector<VectorFieldImageType::Pointer> s_k = VelocitySubtraction(xk[j+1], xk[j]);
                        rho[j] = 1.0f/VelocityProduct(y_k,s_k);
                        alpha[j] = rho[j]*VelocityProduct(s_k,q);
                        q = VelocitySubtraction(q, VelMultiConst(alpha[j],y_k));
                    }
                    
                    
                    std::vector<VectorFieldImageType::Pointer> y_k = VelocitySubtraction(gk[M], gk[M-1]);
                    std::vector<VectorFieldImageType::Pointer> s_k = VelocitySubtraction(xk[M], xk[M-1]);
                    gamma_k = VelocityProduct(s_k, y_k)/VelocityProduct(y_k,y_k);
                    Z = VelMultiConst(gamma_k,q);
                    
                    
                    for(int j=0; j<=M-1; j++)
                    {
                        std::vector<VectorFieldImageType::Pointer> velocitySubtraction = VelocitySubtraction(gk[j + 1], gk[j]);
                        std::vector<VectorFieldImageType::Pointer> subtraction = VelocitySubtraction(xk[j + 1], xk[j]);
                        
                        float beta_j = rho[j]*VelocityProduct(velocitySubtraction, Z);
                        
                        Z = VelocitySubtraction(Z, VelMultiConst( -alpha[j] + beta_j , subtraction));
                    }
                    // update velocity field based on Z
                    /// the followng code is gradient descent for updating v[t]
                   for (int i=0; i<=numberOfTimeIntervals; i++)
                   {
                       // multiply by linear search step size
                       MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                       multiplyByStepSize->SetConstant(scale);
                       multiplyByStepSize->SetInput(Z[i]);
                       multiplyByStepSize->Update();
                       // update v[i]
                       AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                       addUpdateField->SetInput1(velocityField[i]);
                       addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                       addUpdateField->Update();
                       velocityField[i] = addUpdateField->GetOutput();
                   }
                   iter ++;
                   cost_pre = cost_cur;
                }
                
                if (iter<=M)
                {
                    //compute the gradient based on current estimation of v_t
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        // std::cout << "update vector field: " << i << std::endl;
                        // update field = v_t + K*(I(t)grad(lambda(t))), see line 12 in Algorithm 4, in Wei's thesis
                        //compute x,y,Z derivatives of costate image
                        DerivativeFilterType::Pointer       xDerivative = DerivativeFilterType::New();
                        DerivativeFilterType::Pointer       yDerivative = DerivativeFilterType::New();
                        
                        xDerivative->SetInput(costateImages[i]);
                        xDerivative->SetDirection(0);
                        xDerivative->Update();
                        yDerivative->SetInput(costateImages[i]);
                        yDerivative->SetDirection(1);
                        yDerivative->Update();
                        
                        MultiplyImageType::Pointer      multiplyXComponent = MultiplyImageType::New();
                        multiplyXComponent->SetInput1(stateImages[i]);
                        multiplyXComponent->SetInput2(xDerivative->GetOutput());
                        multiplyXComponent->Update();
                        ImageType::Pointer     xProd = multiplyXComponent->GetOutput();
                        
                        MultiplyImageType::Pointer      multiplyYComponent = MultiplyImageType::New();
                        multiplyYComponent->SetInput1(stateImages[i]);
                        multiplyYComponent->SetInput2(yDerivative->GetOutput());
                        multiplyYComponent->Update();
                        ImageType::Pointer     yProd = multiplyYComponent->GetOutput();
                        //   float myKernelSize = gaussianKernelSize*pow(2,(numberOfResolutions-level));
                        float myKernelSize = deformationKernelSizes.at(level-1);
                        
                        GaussianFilterType::Pointer    filterXProd  =  GaussianFilterType::New();
                        filterXProd->SetInput(xProd);
                        filterXProd->SetSigma(myKernelSize);
                        filterXProd->Update();
                        
                        GaussianFilterType::Pointer    filterYProd  =  GaussianFilterType::New();
                        filterYProd->SetInput(yProd);
                        filterYProd->SetSigma(myKernelSize);
                        filterYProd->Update();

                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,filterXProd->GetOutput());
                        composeImageFilter->SetInput(1,filterYProd->GetOutput());
                        composeImageFilter->Update();
                        
                        AddVectorImageType::Pointer     addVectorField  =  AddVectorImageType::New();
                        addVectorField->SetInput1(composeImageFilter->GetOutput());
                        addVectorField->SetInput2(velocityField[i]);
                        addVectorField->Update();

                        updateField[i] = addVectorField->GetOutput();
                    }
                    
                    // keep the past M xk and gk
                    xk[iter - 1] = velocityField;
                    gk[iter - 1] = updateField;
                    
                    // update the velocity field based on the current gradient
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                   {
                       // multiply by linear search step size
                       MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                       multiplyByStepSize->SetConstant(myStepSize);
                       multiplyByStepSize->SetInput(updateField[i]);
                       multiplyByStepSize->Update();
                       // update v[i]
                       AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                       addUpdateField->SetInput1(velocityField[i]);
                       addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                       addUpdateField->Update();
                       velocityField[i] = addUpdateField->GetOutput();
                   }
                   iter++;
                   cost_pre = cost_cur;
                }
            }
        }
        
        //// the following code implements SSD-based geodesic image regression
        /// the computation of forward and backword flow of maps are the same as SSTVD
        /// the computation of state flow, costate flow, update field are different from SSTVD
        if (similarityCostType=="SSD")
        {
           // while (  iter<numberOfIterations.at(level-1) && ( (((cost_pre - cost_cur)/cost_pre)>epsilons.at(level-1)) || iter<=20) && abs(myStepSize)>0.000001 )
            while (  iter<numberOfIterations.at(level-1) && ( convergeRate > epsilons.at(level-1) || iter<=2) && abs(myStepSize)>0.00001 )
            {
                /// solve map flow forward
                for(int l=0; l< numberOfInputs - 1; l++)
                {
                    float intervalLength = intervalLengths[l];
                    for(int i = inputImageTimes.at(l); i< inputImageTimes.at(l+1); i++)
                    {
						// deform each component of v[i] by u[i], u[i+1] = u[i] + v[i](Id + u[i])
						DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
						dispTransform->SetDisplacementField(dispFieldLagrangian[i]);
                    
						// select the x and y compoments of velocityField[i]
						IndexSelectionType::Pointer indexSelectionFilterX = IndexSelectionType::New();
						indexSelectionFilterX->SetIndex(0); // 0 for x direction
						indexSelectionFilterX->SetInput(velocityField[i]);
						indexSelectionFilterX->Update();
                    
						IndexSelectionType::Pointer indexSelectionFilterY = IndexSelectionType::New();
						indexSelectionFilterY->SetIndex(1); // 1 for y direction
						indexSelectionFilterY->SetInput(velocityField[i]);
						indexSelectionFilterY->Update();
                    
						//// deform x,y component of v[i] by u[i]
						ResampleImageFilterType::Pointer   resampleX = ResampleImageFilterType::New();
						resampleX->SetInput(indexSelectionFilterX->GetOutput());
						resampleX->SetSize( imageSizeDown );
						resampleX->SetOutputOrigin(  inputImages[0]->GetOrigin() );
						resampleX->SetOutputSpacing( imageSpacingDown );
						resampleX->SetOutputDirection( inputImages[0]->GetDirection() );
						resampleX->SetTransform(dispTransform);
						resampleX->SetDefaultPixelValue(0);
						resampleX->Update();
                    
						ResampleImageFilterType::Pointer   resampleY = ResampleImageFilterType::New();
						resampleY->SetInput(indexSelectionFilterY->GetOutput());
						resampleY->SetSize( imageSizeDown );
						resampleY->SetOutputOrigin(  inputImages[0]->GetOrigin() );
						resampleY->SetOutputSpacing( imageSpacingDown );
						resampleY->SetOutputDirection( inputImages[0]->GetDirection() );
						resampleY->SetTransform(dispTransform);
						resampleY->SetDefaultPixelValue(0);
						resampleY->Update();
                    
						MultiplyByConstantType::Pointer   multiplyXByIntervalLength  =  MultiplyByConstantType::New();
						multiplyXByIntervalLength->SetConstant(intervalLength);
						multiplyXByIntervalLength->SetInput(resampleX->GetOutput());
						multiplyXByIntervalLength->Update();
                    
						MultiplyByConstantType::Pointer   multiplyYByIntervalLength  =  MultiplyByConstantType::New();
						multiplyYByIntervalLength->SetConstant(intervalLength);
						multiplyYByIntervalLength->SetInput(resampleY->GetOutput());
						multiplyYByIntervalLength->Update();
                    
						ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
						composeImageFilter->SetInput(0,multiplyXByIntervalLength->GetOutput());
						composeImageFilter->SetInput(1,multiplyYByIntervalLength->GetOutput());
						composeImageFilter->Update();
                    
						// u[i+1] = u[i] + v[i](x+u[i])*interval length
						AddVectorImageType::Pointer    addDispFields  = AddVectorImageType::New();
						addDispFields->SetInput1(dispFieldLagrangian[i]);
						addDispFields->SetInput2(composeImageFilter->GetOutput());
						addDispFields->Update();
						dispFieldLagrangian[i+1] = addDispFields->GetOutput();
					}
                }
                //// solve inverse map, we are making assumption that we can compute D\phi[i],this code will fail for large curve flows
                /// this implements Equation (5.15) in Wei Shao's PhD Dissertation
                for(int l=0; l< numberOfInputs - 1; l++)
                {
                    float intervalLength = intervalLengths[l];
                    for(int i = inputImageTimes.at(l); i< inputImageTimes.at(l+1); i++)
                    {
                        // select the x,y compoments of velocityField[i]
                        IndexSelectionType::Pointer VX = IndexSelectionType::New();
                        VX->SetIndex(0); // 0 for x direction
                        VX->SetInput(velocityField[i]);
                        VX->Update();
                        
                        IndexSelectionType::Pointer VY = IndexSelectionType::New();
                        VY->SetIndex(1); // 1 for y direction
                        VY->SetInput(velocityField[i]);
                        VY->Update();
                        
                        // select the x,y,z compoments of phi[i]
                        IndexSelectionType::Pointer UX = IndexSelectionType::New();
                        UX->SetIndex(0); // 0 for x direction
                        UX->SetInput(dispFieldLagrangian[i]);
                        UX->Update();
                        
                        IndexSelectionType::Pointer UY = IndexSelectionType::New();
                        UY->SetIndex(1); // 1 for y direction
                        UY->SetInput(dispFieldLagrangian[i]);
                        UY->Update();
                        
                        //... begin compute Jacobian of phi[i]
                        DerivativeFilterType::Pointer       UXxDerivative = DerivativeFilterType::New();
                        UXxDerivative->SetInput(UX->GetOutput());
                        UXxDerivative->SetDirection(0);
                        UXxDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UXyDerivative = DerivativeFilterType::New();
                        UXyDerivative->SetInput(UX->GetOutput());
                        UXyDerivative->SetDirection(1);
                        UXyDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UYxDerivative = DerivativeFilterType::New();
                        UYxDerivative->SetInput(UY->GetOutput());
                        UYxDerivative->SetDirection(0);
                        UYxDerivative->Update();
                        
                        DerivativeFilterType::Pointer       UYyDerivative = DerivativeFilterType::New();
                        UYyDerivative->SetInput(UY->GetOutput());
                        UYyDerivative->SetDirection(1);
                        UYyDerivative->Update();
                        
                        //... end compute Jacobian of phi[i]
                        
                        
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                        
                        //... begin deform each component of Dphi[i] by phi^{-1}[i]
                        ResampleImageFilterType::Pointer   resample11 = ResampleImageFilterType::New();
                        resample11->SetInput(UXxDerivative->GetOutput());
                        resample11->SetSize( imageSizeDown );
                        resample11->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample11->SetOutputSpacing( imageSpacingDown );
                        resample11->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample11->SetTransform(dispTransform);
                        resample11->SetDefaultPixelValue(0);
                        resample11->Update();
                        ImageType::Pointer D11 = resample11->GetOutput();
                        
                        ResampleImageFilterType::Pointer   resample12 = ResampleImageFilterType::New();
                        resample12->SetInput(UXyDerivative->GetOutput());
                        resample12->SetSize( imageSizeDown );
                        resample12->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample12->SetOutputSpacing( imageSpacingDown );
                        resample12->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample12->SetTransform(dispTransform);
                        resample12->SetDefaultPixelValue(0);
                        resample12->Update();
                        ImageType::Pointer D12 = resample12->GetOutput();
                        

                        ResampleImageFilterType::Pointer   resample21 = ResampleImageFilterType::New();
                        resample21->SetInput(UYxDerivative->GetOutput());
                        resample21->SetSize( imageSizeDown );
                        resample21->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample21->SetOutputSpacing( imageSpacingDown );
                        resample21->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample21->SetTransform(dispTransform);
                        resample21->SetDefaultPixelValue(0);
                        resample21->Update();
                        ImageType::Pointer D21 = resample21->GetOutput();
                        
                        ResampleImageFilterType::Pointer   resample22 = ResampleImageFilterType::New();
                        resample22->SetInput(UYyDerivative->GetOutput());
                        resample22->SetSize( imageSizeDown );
                        resample22->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resample22->SetOutputSpacing( imageSpacingDown );
                        resample22->SetOutputDirection( inputImages[0]->GetDirection() );
                        resample22->SetTransform(dispTransform);
                        resample22->SetDefaultPixelValue(0);
                        resample22->Update();
                        ImageType::Pointer D22 = resample22->GetOutput();
                        // .....end deform components of Dphi[i]
                        
                        //... compute the inverse of Jacobian matrix Dphi[i]\circ\phi^{-1}[i]
                        
                        itk::ImageRegionIterator<ImageType> iterator11(D11,D11->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator12(D12,D12->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator21(D21,D21->GetBufferedRegion());
                        itk::ImageRegionIterator<ImageType> iterator22(D22,D22->GetBufferedRegion());
						
                        iterator11.GoToBegin();
                        iterator12.GoToBegin();
                        iterator21.GoToBegin();
                        iterator22.GoToBegin();

                        while(!iterator11.IsAtEnd())
                        {
                            // Get the current pixel
                            float a = 1+ iterator11.Get();
                            float b = iterator12.Get();
                            float c = iterator21.Get();
                            float d = 1+ iterator22.Get();
                           
                            float det = a*d - b*c;
                            
                            iterator11.Set(d/det);
                            iterator12.Set(-b/det);

                            
                            iterator21.Set(-c/det);
                            iterator22.Set(a/det);

                            ++iterator11;
                            ++iterator12;
                            ++iterator21;
                            ++iterator22;
                        }
                        
                        //.. end
                        
                        /// begin multiply the Jacobian matrix by v[i]
                        MultiplyImageType::Pointer         multiplyXX =  MultiplyImageType::New();
                        multiplyXX->SetInput1(D11);
                        multiplyXX->SetInput2(VX->GetOutput());
                        multiplyXX->Update();
                        
                        MultiplyImageType::Pointer         multiplyXY =  MultiplyImageType::New();
                        multiplyXY->SetInput1(D12);
                        multiplyXY->SetInput2(VY->GetOutput());
                        multiplyXY->Update();
                        
                        
                        AddImageType::Pointer    addX1 = AddImageType::New();
                        addX1->SetInput1(multiplyXX->GetOutput());
                        addX1->SetInput2(multiplyXY->GetOutput());
                        addX1->Update();
                               
                        
                        MultiplyImageType::Pointer         multiplyYX =  MultiplyImageType::New();
                        multiplyYX->SetInput1(D21);
                        multiplyYX->SetInput2(VX->GetOutput());
                        multiplyYX->Update();
                        
                        MultiplyImageType::Pointer         multiplyYY =  MultiplyImageType::New();
                        multiplyYY->SetInput1(D22);
                        multiplyYY->SetInput2(VY->GetOutput());
                        multiplyYY->Update();
                        
                        
                        AddImageType::Pointer    addY1 = AddImageType::New();
                        addY1->SetInput1(multiplyYX->GetOutput());
                        addY1->SetInput2(multiplyYY->GetOutput());
                        addY1->Update();
                        
                        /// end multiply the Jacobian matrix by v[i]
                        
                        /// multiply by -interval length
                        MultiplyByConstantType::Pointer   multiplyXByIntervalLength  =  MultiplyByConstantType::New();
                        multiplyXByIntervalLength->SetConstant(-intervalLength);
                        multiplyXByIntervalLength->SetInput(addX1->GetOutput());
                        multiplyXByIntervalLength->Update();
                        
                        MultiplyByConstantType::Pointer   multiplyYByIntervalLength  =  MultiplyByConstantType::New();
                        multiplyYByIntervalLength->SetConstant(-intervalLength);
                        multiplyYByIntervalLength->SetInput(addY1->GetOutput());
                        multiplyYByIntervalLength->Update();
                        
                        
                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,multiplyXByIntervalLength->GetOutput());
                        composeImageFilter->SetInput(1,multiplyYByIntervalLength->GetOutput());
                        composeImageFilter->Update();
                        
                        // phi^{-1}[i+1] = phi^{-1}[i] + Delta_t * ((-1)Dphi[i]\circ\phi^{-1}[i])^{-1}v[i]
                        AddVectorImageType::Pointer    addDispFields  = AddVectorImageType::New();
                        addDispFields->SetInput1(dispFieldEulerian[i]);
                        addDispFields->SetInput2(composeImageFilter->GetOutput());
                        addDispFields->Update();
                        dispFieldEulerian[i+1] = addDispFields->GetOutput();
                    }
                }
                /// end computation of inverse flow
                        
                
                /// initialize the template image, same size as the downsampled CT
                ImageType::Pointer   templateImage = ImageType::New();
                templateImage->SetBufferedRegion(imageRegionDown);
                templateImage->SetSpacing(imageSpacingDown);
                templateImage->SetDirection(inputImages[0]->GetDirection());
                templateImage->SetOrigin(inputImages[0]->GetOrigin());
                templateImage->Allocate();
                
                /// weight involved in computation of the template
                ImageType::Pointer   weightImage = ImageType::New();
                weightImage->SetBufferedRegion(imageRegionDown);
                weightImage->SetSpacing(imageSpacingDown);
                weightImage->SetDirection(inputImages[0]->GetDirection());
                weightImage->SetOrigin(inputImages[0]->GetOrigin());
                weightImage->Allocate();
                
                itk::ImageRegionIterator<ImageType> templateImageIterator(templateImage,templateImage->GetBufferedRegion());
                while(!templateImageIterator.IsAtEnd())
                {
                    // Set the current pixel
                    templateImageIterator.Set(0.0);
                    ++templateImageIterator;
                }
                
                itk::ImageRegionIterator<ImageType> weightImageIterator(weightImage,weightImage->GetBufferedRegion());
                while(!weightImageIterator.IsAtEnd())
                {
                    // Set the current pixel
                    weightImageIterator.Set(0.0);
                    ++weightImageIterator;
                }
                
                //// if fixedTemplate = false, update template image at each iteration
                if (fixedTemplate=="False")
                {
                    for (int i=0; i<numberOfInputs; i++)
                    {
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(i)]);
                        
                        MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                        multiplyByArtifactMask->SetInput1(inputImagesDown[i]);
                        multiplyByArtifactMask->SetInput2(inputArtifactMasksDown[i]);
                        multiplyByArtifactMask->Update();
                        
                        
                        ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                        resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                        resampleI->SetSize( imageSizeDown );
                        resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleI->SetOutputSpacing( imageSpacingDown );
                        resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleI->SetTransform(dispTransform);
                        resampleI->SetDefaultPixelValue(0);
                        resampleI->Update();
                        
                        /// extra for SSD, compute |D\phit_t_i|
                        JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                        JacobianImage->SetInput(dispFieldLagrangian[inputImageTimes.at(i)]);
                        JacobianImage->Update();
                        
                        // extra for SSD, multiply by |D\phit_t_i|
                        MultiplyImageType::Pointer         multiplyJacobian =  MultiplyImageType::New();
                        multiplyJacobian->SetInput1(resampleI->GetOutput());
                        multiplyJacobian->SetInput2(JacobianImage->GetOutput());
                        multiplyJacobian->Update();
                        
                        // extra for SSD
                        AddImageType::Pointer    addImage = AddImageType::New();
                        addImage->SetInput1(templateImage);
                        addImage->SetInput2(multiplyJacobian->GetOutput());
                        addImage->Update();
                        
                        templateImage = addImage->GetOutput();
                    }
                    
                    
                    for (int i=0; i<numberOfInputs; i++)
                    {
                        // different for SSD
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(i)]);
                        
                        //different for SSD
                        ResampleImageFilterType::Pointer   resampleMask = ResampleImageFilterType::New();
                        resampleMask->SetInput(inputArtifactMasksDown[i]);
                        resampleMask->SetSize( imageSizeDown );
                        resampleMask->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleMask->SetOutputSpacing( imageSpacingDown );
                        resampleMask->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleMask->SetTransform(dispTransform);
                        resampleMask->SetDefaultPixelValue(0);
                        resampleMask->Update();
                        
                        // differenct for SSD
                        JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                        JacobianImage->SetInput(dispFieldLagrangian[inputImageTimes.at(i)]);
                        JacobianImage->Update();
                        
                        // differenct for SSD
                        MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                        multiplyByArtifactMask->SetInput1(JacobianImage->GetOutput());
                        multiplyByArtifactMask->SetInput2(resampleMask->GetOutput());
                        multiplyByArtifactMask->Update();
                        
                        // differenct for SSD
                        AddImageType::Pointer    addImage = AddImageType::New();
                        addImage->SetInput1(weightImage);
                        addImage->SetInput2(multiplyByArtifactMask->GetOutput());
                        addImage->Update();
                        weightImage = addImage->GetOutput();
                    }
                    
                //    DivideImageFilterType::Pointer divideImageFilter = DivideImageFilterType::New ();
                //    divideImageFilter->SetInput1(templateImage);
                 //   divideImageFilter->SetInput2(weightImage);
                 //   divideImageFilter->Update();
                 //   templateImage = divideImageFilter-> GetOutput();
                    templateImage = DivideImage(templateImage,weightImage);
                }
                else
                {
                    templateImage = inputImagesDown[0];
                }
                
                
                //// compute flow of state images, no need to adjust intensity
                
                for(int i=0; i<= numberOfTimeIntervals; i++)
                {
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                    // compute I_0(\phi_t^-1)
                    ResampleImageFilterType::Pointer   resampleI0 = ResampleImageFilterType::New();
                    // resampleI0->SetInput(inputImagesDown[0]);
                    resampleI0->SetInput(templateImage);
                    resampleI0->SetSize( imageSizeDown );
                    resampleI0->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleI0->SetOutputSpacing( imageSpacingDown );
                    resampleI0->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleI0->SetTransform(dispTransform);
                    resampleI0->SetDefaultPixelValue(0);
                    resampleI0->Update();
                    
                    // different for SSD
                    stateImages[i] = resampleI0->GetOutput();
                }
                
                
                float SSD = 0.0;
                float count = 0;
                
                float temp;
                
                //// compute image similarity cost, sum up for I0,I1,I2,...,IN-1
                for (int i=0; i<numberOfInputs; i++)
                {
                    SubtractImageType::Pointer    getDiffImage  =  SubtractImageType::New();
                    getDiffImage->SetInput1(stateImages[inputImageTimes.at(i)]);
                    getDiffImage->SetInput2(inputImagesDown[i]);
                    getDiffImage->Update();
                    
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask1 =  MultiplyImageType::New();
                    multiplyByArtifactMask1->SetInput1(getDiffImage->GetOutput());
                    multiplyByArtifactMask1->SetInput2(inputArtifactMasksDown[i]);
                    multiplyByArtifactMask1->Update();
                    
                    MultiplyImageType::Pointer     squareDiffImage =  MultiplyImageType::New();
                    squareDiffImage->SetInput1(multiplyByArtifactMask1->GetOutput());
                    squareDiffImage->SetInput2(multiplyByArtifactMask1->GetOutput());
                    squareDiffImage->Update();
                    ImageType::Pointer    squareImage = squareDiffImage->GetOutput();
                    
                    itk::ImageRegionIterator<ImageType> imageIterator(squareImage,squareImage->GetBufferedRegion());
                    while(!imageIterator.IsAtEnd())
                    {
                        // Set the current pixel
                        temp = imageIterator.Get();
                        if (temp>0.01) // only compute SSD in the lung
                        {
                            SSD = SSD + temp;
                            count++;
                        }
                        ++imageIterator;
                    }
                    
                }
                
                cost_cur = SSD/count; // compute mean SSD over all input images
                
                /// output loss, stepsize at current iteration
                if (iter<=1)
                {
                    std::cout << "Iteration number=" << iter << ", Similarity Cost=" << SSD/count << ", Reduction rate=" << ", Step size=" << -1.0*myStepSize << std::endl;
                    file << "Iteration number=" + to_string(iter) + ", Similarity Cost=" + to_string(SSD/count) + ", Reduction rate= " + ", step size= " + to_string(-1.0*myStepSize) + "\n";
                }
                else
                {
                    std::cout << "Iteration number=" << iter << ", Similarity Cost=" << SSD/count << ", Reduction rate=" << (cost_pre - cost_cur)/cost_pre << ", Step size=" << -1.0*myStepSize <<std::endl;
                    file << "Iteration number=" + to_string(iter) + ", Similarity Cost=" + to_string(SSD/count) + ", Reduction rate=" + to_string((cost_pre - cost_cur)/cost_pre) + " step size=" + to_string(-1.0*myStepSize) + "\n";
                }
                
                // get lambda[t_N] = (2/(sigma^2)(I(1) - I_1))*M_1 (costate)
                // this implements line 5 in Algorithm 4 in Wei's Phd thesis
                SubtractImageType::Pointer    getCostateError  =  SubtractImageType::New();
                getCostateError->SetInput1(stateImages[numberOfTimeIntervals]);
                getCostateError->SetInput2(inputImagesDown[numberOfInputs-1]);
                getCostateError->Update();
                
                MultiplyByConstantType::Pointer  multiplyByWeight =  MultiplyByConstantType::New();
                multiplyByWeight->SetConstant(2.0f/(imageWeights.at(level-1)*imageWeights.at(level-1)));
                multiplyByWeight->SetInput(getCostateError->GetOutput());
                multiplyByWeight->Update();
                costateImages[numberOfTimeIntervals] = multiplyByWeight-> GetOutput();
                
                MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                multiplyByArtifactMask->SetInput1(costateImages[numberOfTimeIntervals]);
                multiplyByArtifactMask->SetInput2(inputArtifactMasksDown[numberOfInputs-1]);
                multiplyByArtifactMask->Update();
                costateImages[numberOfTimeIntervals] = multiplyByArtifactMask->GetOutput();
                
                /////// compute flow of costates, different for SSD
                
                ImageType::Pointer      startImage = costateImages[numberOfTimeIntervals];
                
                int k = numberOfInputs-1;
                while (k>0) {
                    
                    for(int i = inputImageTimes.at(k-1); i< inputImageTimes.at(k); i++)
                    {
                        DisplacementTransformType::Pointer    dispTransformPhik = DisplacementTransformType::New();
                        dispTransformPhik->SetDisplacementField(dispFieldLagrangian[inputImageTimes.at(k)]);
                        
                        ResampleImageFilterType::Pointer   resampleLambda = ResampleImageFilterType::New();
                        resampleLambda->SetInput(startImage);
                        resampleLambda->SetSize( imageSizeDown );
                        resampleLambda->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleLambda->SetOutputSpacing( imageSpacingDown );
                        resampleLambda->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleLambda->SetTransform(dispTransformPhik);
                        resampleLambda->SetDefaultPixelValue(0);
                        resampleLambda->Update();
                        
                        ///// different for SSD
                        ////....start estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                        JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                        JacobianImage->SetInput(dispFieldLagrangian[inputImageTimes.at(k)]);
                        JacobianImage->Update();
                        
                        DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
                        defJac->SetDisplacementField(dispFieldEulerian[inputImageTimes.at(k)]);
                        
                        ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
                        resampleJac->SetInput(JacobianImage->GetOutput());
                        resampleJac->SetSize( imageSizeDown );
                        resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleJac->SetOutputSpacing( imageSpacingDown );
                        resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleJac->SetTransform(defJac);
                        resampleJac->SetDefaultPixelValue(1);
                        resampleJac->Update();
                        
                        ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
                        
                        itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
                        imageIterator.GoToBegin();
                        while(!imageIterator.IsAtEnd())
                        {
                            // Get the current pixel
                            float n = imageIterator.Get();
                            imageIterator.Set(1/n);
                            ++imageIterator;
                        }
                        ////....finish estimate Jacobian of phi[k]^{-1} by Jacobian of phi[k]
                        
                        
                        MultiplyImageType::Pointer         multiplyJacobian =  MultiplyImageType::New();
                        multiplyJacobian->SetInput1(resampleLambda->GetOutput());
                        multiplyJacobian->SetInput2(myJacobian);
                        multiplyJacobian->Update();
                        
                        
                        DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                        dispTransform->SetDisplacementField(dispFieldEulerian[i]);
                        
                        /// resample by phi_t^{-1}
                        
                        ResampleImageFilterType::Pointer   resampleImage = ResampleImageFilterType::New();
                        resampleImage->SetInput(multiplyJacobian->GetOutput());
                        resampleImage->SetSize( imageSizeDown );
                        resampleImage->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                        resampleImage->SetOutputSpacing( imageSpacingDown );
                        resampleImage->SetOutputDirection( inputImages[0]->GetDirection() );
                        resampleImage->SetTransform(dispTransform);
                        resampleImage->SetDefaultPixelValue(0);
                        resampleImage->Update();
                        
                        
                        ///// different for SSD
                        JacobianFilterType::Pointer         JacobianImagePhi = JacobianFilterType::New();
                        JacobianImagePhi->SetInput(dispFieldEulerian[i]);
                        JacobianImagePhi->Update();
                        
                        MultiplyImageType::Pointer         multiplyJacobianPhi =  MultiplyImageType::New();
                        multiplyJacobianPhi->SetInput1(resampleImage->GetOutput());
                        multiplyJacobianPhi->SetInput2(JacobianImagePhi->GetOutput());
                        multiplyJacobianPhi->Update();
                        
                        
                        costateImages[i] =  multiplyJacobianPhi->GetOutput();
                    }
                    k=k-1;
                    
                    //// compute lambda(ti-) = lambda(ti-) + 2/(sigma^2)(I(t_i) - Ii)
                    // get lambda[t_N] = 2/(sigma^2)(I(1) - I_1)
                    SubtractImageType::Pointer    costateError  =  SubtractImageType::New();
                    costateError->SetInput1(stateImages[inputImageTimes.at(k)]);
                    costateError->SetInput2(inputImagesDown[k]);
                    costateError->Update();
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask2 =  MultiplyImageType::New();
                    multiplyByArtifactMask2->SetInput1(costateError->GetOutput());
                    multiplyByArtifactMask2->SetInput2(inputArtifactMasksDown[k]);
                    multiplyByArtifactMask2->Update();
                    
                    MultiplyByConstantType::Pointer  weight =  MultiplyByConstantType::New();
                    weight->SetConstant(2.0f / (imageWeights.at(level - 1) * imageWeights.at(level - 1)));
                    weight->SetInput(multiplyByArtifactMask2->GetOutput());
                    weight->Update();
                    
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(weight->GetOutput());
                    addImage->SetInput2(costateImages[inputImageTimes.at(k)]);
                    addImage->Update();
                    
                    startImage = addImage->GetOutput();
                }
                
                
                
                //// compute cost reduction rate
                if(cost_cur<cost_pre)
                {
                    convergeRate = (cost_pre - cost_cur)/cost_pre;
                }
                

                //// Update Velocity Field by gradient descent
                /// v_t = v_t - s* (v_t - K*(lamdba(t)Grad(I(t))))
                /// We will Gaussian filter to each compnengt of lamdba(t)Grad(I(t))
                
                if (cost_cur<cost_pre||iter==1)
                {
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        // std::cout << "update vector field: " << i << std::endl;
                        // update field = v_t + K*(lambda(t)grad(I(t))), see line 12 in Algorithm 5, in Wei's thesis
                        //compute x,y,Z derivatives of costate image
                        
                        DerivativeFilterType::Pointer       xDerivative = DerivativeFilterType::New();
                        DerivativeFilterType::Pointer       yDerivative = DerivativeFilterType::New();
                        
                        xDerivative->SetInput(stateImages[i]);
                        xDerivative->SetDirection(0);
                        xDerivative->Update();
                        yDerivative->SetInput(stateImages[i]);
                        yDerivative->SetDirection(1);
                        yDerivative->Update();

                        MultiplyImageType::Pointer      multiplyXComponent = MultiplyImageType::New();
                        multiplyXComponent->SetInput1(costateImages[i]);
                        multiplyXComponent->SetInput2(xDerivative->GetOutput());
                        multiplyXComponent->Update();
                        ImageType::Pointer     xProd = multiplyXComponent->GetOutput();
                        
                        MultiplyImageType::Pointer      multiplyYComponent = MultiplyImageType::New();
                        multiplyYComponent->SetInput1(costateImages[i]);
                        multiplyYComponent->SetInput2(yDerivative->GetOutput());
                        multiplyYComponent->Update();
                        ImageType::Pointer     yProd = multiplyYComponent->GetOutput();
                        
                        
                        
                        //   float myKernelSize = gaussianKernelSize*pow(2,(numberOfResolutions-level));
                        float myKernelSize = deformationKernelSizes.at(level-1);
                        
                        GaussianFilterType::Pointer    filterXProd  =  GaussianFilterType::New();
                        filterXProd->SetInput(xProd);
                        filterXProd->SetSigma(myKernelSize);
                        filterXProd->Update();
                        
                        GaussianFilterType::Pointer    filterYProd  =  GaussianFilterType::New();
                        filterYProd->SetInput(yProd);
                        filterYProd->SetSigma(myKernelSize);
                        filterYProd->Update();
                        
                        
                        ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
                        composeImageFilter->SetInput(0,filterXProd->GetOutput());
                        composeImageFilter->SetInput(1,filterYProd->GetOutput());
                        composeImageFilter->Update();
                        
                        
                        SubtractVectorImageType::Pointer      subtractVectorField = SubtractVectorImageType::New();
                        subtractVectorField->SetInput1(velocityField[i]);
                        subtractVectorField->SetInput2(composeImageFilter->GetOutput());
                        subtractVectorField->Update();
                        
                        updateField[i] = subtractVectorField->GetOutput();
                        
                    
                    }
                    
                    /// the followng code is gradient descent for updating v[t]
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        /// choose smaller step size for higher resolution
                        
                        // std::cout<< "my step size: "<< myStepSize << std::endl;
                        
                        // multiply by linear search step size
                        MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                        multiplyByStepSize->SetConstant(myStepSize);
                        multiplyByStepSize->SetInput(updateField[i]);
                        multiplyByStepSize->Update();
                        
                        
                        // update v[i]
                        AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                        addUpdateField->SetInput1(velocityField[i]);
                        addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                        addUpdateField->Update();
                        velocityField[i] = addUpdateField->GetOutput();
                    }
                    iter++;
                    cost_pre = cost_cur;
                }
                else
                {
                    /// if the cost is not decreasing, we need to choose a smaller step size
                    myStepSize = myStepSize*0.5f;
                    for (int i=0; i<=numberOfTimeIntervals; i++)
                    {
                        /// choose smaller step size for higher resolution
                        
                        // std::cout<< "my step size: "<< myStepSize << std::endl;
                        
                        // multiply by linear search step size
                        MultiplyVectorImageByConstantType::Pointer   multiplyByStepSize  =  MultiplyVectorImageByConstantType::New();
                        multiplyByStepSize->SetConstant(-myStepSize);
                        multiplyByStepSize->SetInput(updateField[i]);
                        multiplyByStepSize->Update();
                        
                        
                        // update v[i]
                        AddVectorImageType::Pointer   addUpdateField  = AddVectorImageType::New();
                        addUpdateField->SetInput1(velocityField[i]);
                        addUpdateField->SetInput2(multiplyByStepSize->GetOutput());
                        addUpdateField->Update();
                        velocityField[i] = addUpdateField->GetOutput();
                    }
                }
            }
            
            
        }
        
        //// ... done with CT intensity image regression
        
        
        // output velocity field
  //      for(int i =0; i < numberOfInputs; i++)
  //      {
  //          VectorImageWriterType::Pointer   vectorImageWriter =  VectorImageWriterType::New();
  //          vectorImageWriter->SetFileName(outputDirectory + "vector_field_" + std::to_string(inputImageTimes.at(i)) + "_Res_" + std::to_string(level) + ".nii");
  //          vectorImageWriter->SetInput(velocityField[inputImageTimes.at(i)]);
  //          vectorImageWriter->Update();
  //      }
        
        // output state images
  //      for (int i=0; i< numberOfInputs; i++)
 //       {
 //          ImageWriterType::Pointer ImageWriter0 = ImageWriterType::New();
 //          ImageWriter0->SetFileName(outputDirectory + "state_time_" + std::to_string(inputImageTimes.at(i)) + "_Res_" + std::to_string(level) + ".nii.gz");
 //           ImageWriter0->SetInput(stateImages[inputImageTimes.at(i)]);
  //          ImageWriter0->Update();
 //       }
   //
         // output costate images
 //       for (int i=0; i< numberOfInputs; i++)
 //       {
 //           ImageWriterType::Pointer ImageWriter0 = ImageWriterType::New();
 //           ImageWriter0->SetFileName(outputDirectory + "costate_time_" + std::to_string(inputImageTimes.at(i)) + "_Res_" + std::to_string(level) + ".nii");
 //           ImageWriter0->SetInput(costateImages[inputImageTimes.at(i)]);
  //          ImageWriter0->Update();
  //      }
        
         // output downsampled input images
  //      for (int i=0; i<numberOfInputs; i++)
  //      {
  //          ImageWriterType::Pointer ImageWriter = ImageWriterType::New();
  //          ImageWriter->SetFileName(outputDirectory + "input_image_at_time_" + std::to_string(inputImageTimes.at(i)) + "_Res_" + std::to_string(level) + ".nii.gz");
  //          ImageWriter->SetInput(inputImagesDown[i]);
  //          ImageWriter->Update();
  //      }
        
        
        //...  begin upsample velocity field for next resolution
        if (level < numberOfResolutions)
        {
            ImageType::RegionType imageRegionUp;
            ImageType::SizeType imageSizeUp = imageSize;
            ImageType::SpacingType imageSpacingUp = imageSpacing;
            int upSampleFactor = downsampleFactors.at(level);
            // int upSampleFactor = 1;
            imageSizeUp[0] = imageSizeUp[0]/upSampleFactor;
            imageSizeUp[1] = imageSizeUp[1]/upSampleFactor;

            imageSpacingUp[0] = imageSpacingUp[0]*upSampleFactor;
            imageSpacingUp[1] = imageSpacingUp[1]*upSampleFactor;

            imageRegionUp.SetSize(imageSizeUp);
            
            for(int i =0; i <= numberOfTimeIntervals; i++)
            {
                ResampleVectorImageFilterType::Pointer   resampleVelocity = ResampleVectorImageFilterType::New();
                resampleVelocity->SetSize( imageSizeUp);
                resampleVelocity->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                resampleVelocity->SetOutputSpacing( imageSpacingUp);
                resampleVelocity->SetOutputDirection( inputImages[0]->GetDirection() );
                resampleVelocity->SetInput(velocityField[i]);
                resampleVelocity->Update();
                velocityField[i] = resampleVelocity->GetOutput();
            }
        }
         //...  end upsample velocity field for next resolution
        
        
        /// compute running time for current resolution
        auto finish = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<float> elapsed = finish - start;
        regTimes[level-1] = elapsed.count(); // get execution time in seconds
        int hour = regTimes[level-1]/3600;
        int minute = (regTimes[level-1] - hour*3600)/60;
        int second = regTimes[level-1] - hour*3600 - minute*60;
        /// output running time in the format of xx hours, xx minutes, xx seconds
		
        cout << "time spent on resolution " << level << " is " << hour << "h" << minute << "m" << second <<  "s\n";
        
        file << "time spent on resolution " + to_string(level) + " is " + to_string(hour) + "h" + to_string(minute) + "m" + to_string(second) + "s\n";
        
        /// once the regression is done, free memory for the following images
        if (level==numberOfResolutions)
        {
            dispFieldEulerian.clear();
            dispFieldLagrangian.clear();
            inputImagesDown.clear();
            inputArtifactMasksDown.clear();
            stateImages.clear();
            costateImages.clear();
            updateField.clear();
        }
        
    }
    ////........ end of for loop of regression procession
    cout << "---------------------------------------" << endl;
    cout << "summary:" << endl;
    
    int totalRunningTime = 0; // calculate total running time in seconds
    for (int level=1; level<=numberOfResolutions; level++)
    {
        int hour = regTimes[level-1]/3600;
        int minute = (regTimes[level-1] - hour*3600)/60;
        int second = regTimes[level-1] - hour*3600 - minute*60;
        cout << "time spent on resolution " << level << " is " << hour << "h" << minute << "m" << second <<  "s\n";
        file << "time spent on resolution " + to_string(level) + " is " + to_string(hour) + "h" + to_string(minute) + "m" + to_string(second) + "s\n";
        
        totalRunningTime =  totalRunningTime + regTimes[level-1];
    }
    int hour = totalRunningTime/3600;
    int minute = (totalRunningTime - hour*3600)/60;
    int second = totalRunningTime - hour*3600 - minute*60;
    /// output total amount of time spent on regression
    cout << "total running time is " << hour << "h" << minute << "m" << second <<  "s\n";
    file << "total running time is " + to_string(hour) + "h" + to_string(minute) + "m" + to_string(second) + "s\n";
    
    ////////// upsample velocity fields to full resolution
    for(int i =0; i <= numberOfTimeIntervals; i++)
    {
        ResampleVectorImageFilterType::Pointer   resampleVelocity = ResampleVectorImageFilterType::New();
        resampleVelocity->SetSize( imageSize);
        resampleVelocity->SetOutputOrigin(  inputImages[0]->GetOrigin() );
        resampleVelocity->SetOutputSpacing( imageSpacing);
        resampleVelocity->SetOutputDirection( inputImages[0]->GetDirection() );
        resampleVelocity->SetInput(velocityField[i]);
        resampleVelocity->Update();
        velocityField[i] = resampleVelocity->GetOutput();
    }

    
    //// compute pushforward and pull back transformations simultaneously
    std::vector<VectorFieldImageType::Pointer>      dispFieldPush;
    
    // initilalize displacement field
    for(int i =0; i < numberOfInputs; i++)
    {
        VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
        dispFieldPush.emplace_back(displacement);
        displacement->SetBufferedRegion(imageRegion);
        displacement->SetSpacing(imageSpacing);
        displacement->SetDirection(inputImages[0]->GetDirection());
        displacement->SetOrigin(inputImages[0]->GetOrigin());
        displacement->Allocate();
    }
    
    for(int i=0; i< numberOfInputs; i++)
    {
        itk::ImageRegionIterator<VectorFieldImageType> dispFieldIterator(dispFieldPush[i], dispFieldPush[i]->GetBufferedRegion());
        while(!dispFieldIterator.IsAtEnd())
        {
            VectorPixelType v;
            v[0] = 0.0;
            v[1] = 0.0;

            dispFieldIterator.Set(v);
            ++dispFieldIterator;
        }
    }
    
    std::vector<VectorFieldImageType::Pointer>      dispFieldPull;
    
    // initilalize displacement field
    for(int i =0; i < numberOfInputs; i++)
    {
        VectorFieldImageType::Pointer  displacement = VectorFieldImageType::New();
        dispFieldPull.emplace_back(displacement);
        displacement->SetBufferedRegion(imageRegion);
        displacement->SetSpacing(imageSpacing);
        displacement->SetDirection(inputImages[0]->GetDirection());
        displacement->SetOrigin(inputImages[0]->GetOrigin());
        displacement->Allocate();
    }
    
    for(int i=0; i < numberOfInputs; i++)
    {
        itk::ImageRegionIterator<VectorFieldImageType> displacementFieldIterator(dispFieldPull[i], dispFieldPull[i]->GetBufferedRegion());
        while(!displacementFieldIterator.IsAtEnd())
        {
            VectorPixelType v;
            v[0] = 0.0;
            v[1] = 0.0;

            displacementFieldIterator.Set(v);
            ++displacementFieldIterator;
        }
    }
    
    //// the following computation of map flows is not saving all intermediate results, this saves lots of memory
    for(int l=outputDispStartTime; l< numberOfInputs - 1; l++)
    {
        VectorFieldImageType::Pointer  dispCurrentPush = VectorFieldImageType::New();
        dispCurrentPush = dispFieldPush[l];
        
        VectorFieldImageType::Pointer  dispCurrentPull = VectorFieldImageType::New();
        dispCurrentPull = dispFieldPull[l];
        
        float intervalLength = intervalLengths[l];
        
        for(int i = inputImageTimes.at(l); i< inputImageTimes.at(l+1); i++)
        {
            
            // select the x,y,z compoments of velocityField[i]
            IndexSelectionType::Pointer VX = IndexSelectionType::New();
            VX->SetIndex(0); // 0 for x direction
            VX->SetInput(velocityField[i]);
            VX->Update();
            
            IndexSelectionType::Pointer VY = IndexSelectionType::New();
            VY->SetIndex(1); // 1 for y direction
            VY->SetInput(velocityField[i]);
            VY->Update();
            
            // select the x,y,z compoments of phi[i]
            IndexSelectionType::Pointer UX = IndexSelectionType::New();
            UX->SetIndex(0); // 0 for x direction
            UX->SetInput(dispCurrentPush);
            UX->Update();
            
            IndexSelectionType::Pointer UY = IndexSelectionType::New();
            UY->SetIndex(1); // 1 for y direction
            UY->SetInput(dispCurrentPush);
            UY->Update();
            
            //... begin compute Jacobian of phi[i]
            DerivativeFilterType::Pointer       UXxDerivative = DerivativeFilterType::New();
            UXxDerivative->SetInput(UX->GetOutput());
            UXxDerivative->SetDirection(0);
            UXxDerivative->Update();
            
            DerivativeFilterType::Pointer       UXyDerivative = DerivativeFilterType::New();
            UXyDerivative->SetInput(UX->GetOutput());
            UXyDerivative->SetDirection(1);
            UXyDerivative->Update();
            
            DerivativeFilterType::Pointer       UYxDerivative = DerivativeFilterType::New();
            UYxDerivative->SetInput(UY->GetOutput());
            UYxDerivative->SetDirection(0);
            UYxDerivative->Update();
            
            DerivativeFilterType::Pointer       UYyDerivative = DerivativeFilterType::New();
            UYyDerivative->SetInput(UY->GetOutput());
            UYyDerivative->SetDirection(1);
            UYyDerivative->Update();
            //... end compute Jacobian of phi[i]
            
            
            DisplacementTransformType::Pointer    dispTransformEulerian = DisplacementTransformType::New();
            dispTransformEulerian->SetDisplacementField(dispCurrentPull);
            
            //... begin deform each component of Dphi[i] by phi^{-1}[i]
            ResampleImageFilterType::Pointer   resample11 = ResampleImageFilterType::New();
            resample11->SetInput(UXxDerivative->GetOutput());
            resample11->SetSize( imageSize );
            resample11->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resample11->SetOutputSpacing( imageSpacing );
            resample11->SetOutputDirection( inputImages[0]->GetDirection() );
            resample11->SetTransform(dispTransformEulerian);
            resample11->SetDefaultPixelValue(0);
            resample11->Update();
            ImageType::Pointer D11 = resample11->GetOutput();
            
            ResampleImageFilterType::Pointer   resample12 = ResampleImageFilterType::New();
            resample12->SetInput(UXyDerivative->GetOutput());
            resample12->SetSize( imageSize );
            resample12->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resample12->SetOutputSpacing( imageSpacing );
            resample12->SetOutputDirection( inputImages[0]->GetDirection() );
            resample12->SetTransform(dispTransformEulerian);
            resample12->SetDefaultPixelValue(0);
            resample12->Update();
            ImageType::Pointer D12 = resample12->GetOutput();
            
            
            ResampleImageFilterType::Pointer   resample21 = ResampleImageFilterType::New();
            resample21->SetInput(UYxDerivative->GetOutput());
            resample21->SetSize( imageSize);
            resample21->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resample21->SetOutputSpacing( imageSpacing );
            resample21->SetOutputDirection( inputImages[0]->GetDirection() );
            resample21->SetTransform(dispTransformEulerian);
            resample21->SetDefaultPixelValue(0);
            resample21->Update();
            ImageType::Pointer D21 = resample21->GetOutput();
            
            ResampleImageFilterType::Pointer   resample22 = ResampleImageFilterType::New();
            resample22->SetInput(UYyDerivative->GetOutput());
            resample22->SetSize( imageSize );
            resample22->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resample22->SetOutputSpacing( imageSpacing );
            resample22->SetOutputDirection( inputImages[0]->GetDirection() );
            resample22->SetTransform(dispTransformEulerian);
            resample22->SetDefaultPixelValue(0);
            resample22->Update();
            ImageType::Pointer D22 = resample22->GetOutput();
            
            // .....end deform components of Dphi[i]
            
            //... compute the inverse of Jacobian matrix
            
            itk::ImageRegionIterator<ImageType> iterator11(D11,D11->GetBufferedRegion());
            itk::ImageRegionIterator<ImageType> iterator12(D12,D12->GetBufferedRegion());
            itk::ImageRegionIterator<ImageType> iterator21(D21,D21->GetBufferedRegion());
            itk::ImageRegionIterator<ImageType> iterator22(D22,D22->GetBufferedRegion());
			
            iterator11.GoToBegin();
            iterator12.GoToBegin();
            iterator21.GoToBegin();
            iterator22.GoToBegin();

            while(!iterator11.IsAtEnd())
            {
                // Get the current pixel
                float a = 1+ iterator11.Get();
                float b = iterator12.Get();
                float c = iterator21.Get();
                float d = 1+ iterator22.Get();
                
                float det = a*d - b*c;
                
                iterator11.Set(d/det);
                iterator12.Set(-b/det);
                
                iterator21.Set(-c/det);
                iterator22.Set(a/det);

                ++iterator11;
                ++iterator12;
                ++iterator21;
                ++iterator22;
            }
            
            //.. end
            
            /// begin multiply the Jacobian matrix by v[i]
            MultiplyImageType::Pointer         multiplyXX =  MultiplyImageType::New();
            multiplyXX->SetInput1(D11);
            multiplyXX->SetInput2(VX->GetOutput());
            multiplyXX->Update();
            
            MultiplyImageType::Pointer         multiplyXY =  MultiplyImageType::New();
            multiplyXY->SetInput1(D12);
            multiplyXY->SetInput2(VY->GetOutput());
            multiplyXY->Update();
            
            AddImageType::Pointer    addX1 = AddImageType::New();
            addX1->SetInput1(multiplyXX->GetOutput());
            addX1->SetInput2(multiplyXY->GetOutput());
            addX1->Update();
            
            
            MultiplyImageType::Pointer         multiplyYX =  MultiplyImageType::New();
            multiplyYX->SetInput1(D21);
            multiplyYX->SetInput2(VX->GetOutput());
            multiplyYX->Update();
            
            MultiplyImageType::Pointer         multiplyYY =  MultiplyImageType::New();
            multiplyYY->SetInput1(D22);
            multiplyYY->SetInput2(VY->GetOutput());
            multiplyYY->Update();
            
            AddImageType::Pointer    addY1 = AddImageType::New();
            addY1->SetInput1(multiplyYX->GetOutput());
            addY1->SetInput2(multiplyYY->GetOutput());
            addY1->Update();
            /// end multiply the Jacobian matrix by v[i]
            
            /// multiply by -interval length
            MultiplyByConstantType::Pointer   multiplyXByIntervalLengthEulerian  =  MultiplyByConstantType::New();
            multiplyXByIntervalLengthEulerian->SetConstant(-intervalLength);
            multiplyXByIntervalLengthEulerian->SetInput(addX1->GetOutput());
            multiplyXByIntervalLengthEulerian->Update();
            
            MultiplyByConstantType::Pointer   multiplyYByIntervalLengthEulerian  =  MultiplyByConstantType::New();
            multiplyYByIntervalLengthEulerian->SetConstant(-intervalLength);
            multiplyYByIntervalLengthEulerian->SetInput(addY1->GetOutput());
            multiplyYByIntervalLengthEulerian->Update();
            
            
            ImageToVectorImageFilterType::Pointer       composeImageFilterEulerian =  ImageToVectorImageFilterType::New();
            composeImageFilterEulerian->SetInput(0,multiplyXByIntervalLengthEulerian->GetOutput());
            composeImageFilterEulerian->SetInput(1,multiplyYByIntervalLengthEulerian->GetOutput());
            composeImageFilterEulerian->Update();
            
            // u[i+1] = u[i] + v[i](x+u[i])*interval length
            AddVectorImageType::Pointer    addDispFieldsEulerian  = AddVectorImageType::New();
            addDispFieldsEulerian->SetInput1(dispCurrentPull);
            addDispFieldsEulerian->SetInput2(composeImageFilterEulerian->GetOutput());
            addDispFieldsEulerian->Update();
            dispCurrentPull = addDispFieldsEulerian->GetOutput();
            
            
            // deform each component of v[i] by u[i], u[i+1] = u[i] + v[i](Id + u[i])
            DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
            dispTransform->SetDisplacementField(dispCurrentPush);
            
            // select the x and y compoments of velocityField[i]
            IndexSelectionType::Pointer indexSelectionFilterX = IndexSelectionType::New();
            indexSelectionFilterX->SetIndex(0); // 0 for x direction
            indexSelectionFilterX->SetInput(velocityField[i]);
            indexSelectionFilterX->Update();
            
            IndexSelectionType::Pointer indexSelectionFilterY = IndexSelectionType::New();
            indexSelectionFilterY->SetIndex(1); // 1 for y direction
            indexSelectionFilterY->SetInput(velocityField[i]);
            indexSelectionFilterY->Update();
            
            //// deform x,y component of v[i] by u[i]
            ResampleImageFilterType::Pointer   resampleX = ResampleImageFilterType::New();
            resampleX->SetInput(indexSelectionFilterX->GetOutput());
            resampleX->SetSize( imageSize );
            resampleX->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resampleX->SetOutputSpacing( inputImages[0]->GetSpacing() );
            resampleX->SetOutputDirection( inputImages[0]->GetDirection() );
            //  resampleX->SetReferenceImage(I0);
            resampleX->SetTransform(dispTransform);
            resampleX->SetDefaultPixelValue(0);
            resampleX->Update();
            
            ResampleImageFilterType::Pointer   resampleY = ResampleImageFilterType::New();
            resampleY->SetInput(indexSelectionFilterY->GetOutput());
            resampleY->SetSize( imageSize );
            resampleY->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resampleY->SetOutputSpacing( inputImages[0]->GetSpacing() );
            resampleY->SetOutputDirection( inputImages[0]->GetDirection() );
            // resampleY->SetReferenceImage(I0);
            resampleY->SetTransform(dispTransform);
            resampleY->SetDefaultPixelValue(0);
            resampleY->Update();
            
            
            MultiplyByConstantType::Pointer   multiplyXByIntervalLength  =  MultiplyByConstantType::New();
            multiplyXByIntervalLength->SetConstant(intervalLength);
            multiplyXByIntervalLength->SetInput(resampleX->GetOutput());
            multiplyXByIntervalLength->Update();
            
            MultiplyByConstantType::Pointer   multiplyYByIntervalLength  =  MultiplyByConstantType::New();
            multiplyYByIntervalLength->SetConstant(intervalLength);
            multiplyYByIntervalLength->SetInput(resampleY->GetOutput());
            multiplyYByIntervalLength->Update();

            ImageToVectorImageFilterType::Pointer       composeImageFilter =  ImageToVectorImageFilterType::New();
            composeImageFilter->SetInput(0,multiplyXByIntervalLength->GetOutput());
            composeImageFilter->SetInput(1,multiplyYByIntervalLength->GetOutput());
            composeImageFilter->Update();
            
            // u[i+1] = u[i] + v[i](x+u[i])*interval length
            AddVectorImageType::Pointer    addDispFields  = AddVectorImageType::New();
            addDispFields->SetInput1(dispCurrentPush);
            addDispFields->SetInput2(composeImageFilter->GetOutput());
            addDispFields->Update();
            dispCurrentPush = addDispFields->GetOutput();
        }
        dispFieldPush[l+1] = dispCurrentPush;
        dispFieldPull[l+1] = dispCurrentPull;
    }
    
    ///// compute the final template image
    ImageType::Pointer   templateImageFinal = ImageType::New();
    templateImageFinal->SetBufferedRegion(imageRegion);
    templateImageFinal->SetSpacing(imageSpacing);
    templateImageFinal->SetDirection(inputImages[0]->GetDirection());
    templateImageFinal->SetOrigin(inputImages[0]->GetOrigin());
    templateImageFinal->Allocate();
    
    ImageType::Pointer   weightImageFinal  = ImageType::New();
    weightImageFinal ->SetBufferedRegion(imageRegion);
    weightImageFinal ->SetSpacing(imageSpacing);
    weightImageFinal ->SetDirection(inputImages[0]->GetDirection());
    weightImageFinal ->SetOrigin(inputImages[0]->GetOrigin());
    weightImageFinal ->Allocate();
    
    itk::ImageRegionIterator<ImageType> templateImageIterator(templateImageFinal,templateImageFinal->GetBufferedRegion());
    while(!templateImageIterator.IsAtEnd())
    {
        // Set the current pixel
        templateImageIterator.Set(0.0);
        ++templateImageIterator;
    }
    
    itk::ImageRegionIterator<ImageType> weightImageIterator(weightImageFinal,weightImageFinal->GetBufferedRegion());
    while(!weightImageIterator.IsAtEnd())
    {
        // Set the current pixel
        weightImageIterator.Set(0.0);
        ++weightImageIterator;
    }
        
        //// if fixedTemplate = false, update template image at each iteration
        if (fixedTemplate=="False")
        {
            if (similarityCostType=="SSTVD")
            {
                for (int i=0; i<numberOfInputs; i++)
                {
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldPush[i]);
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                    multiplyByArtifactMask->SetInput1(originalInputImages[i]);
                    multiplyByArtifactMask->SetInput2(inputArtifactMasks[i]);
                    multiplyByArtifactMask->Update();
                    
                    
                    ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                    resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                    resampleI->SetSize( imageSize );
                    resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleI->SetOutputSpacing( imageSpacing );
                    resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleI->SetTransform(dispTransform);
                    resampleI->SetDefaultPixelValue(0);
                    resampleI->Update();
                    
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(templateImageFinal);
                    addImage->SetInput2(resampleI->GetOutput());
                    addImage->Update();
                    
                    templateImageFinal = addImage->GetOutput();
                }
                for (int i=0; i<numberOfInputs; i++)
                {
                    
                    
                    
                    //// estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                    JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                    JacobianImage->SetInput(dispFieldPush[i]);
                    JacobianImage->Update();
                    
                    DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
                    defJac->SetDisplacementField(dispFieldPull[i]);
                    
                    ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
                    resampleJac->SetInput(JacobianImage->GetOutput());
                    resampleJac->SetSize( imageSize );
                    resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleJac->SetOutputSpacing( imageSpacing );
                    resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleJac->SetTransform(defJac);
                    resampleJac->SetDefaultPixelValue(1);
                    resampleJac->Update();
                    
                    ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
                    
                    
                    itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
                    imageIterator.GoToBegin();
                    while(!imageIterator.IsAtEnd())
                    {
                        // Get the current pixel
                        float n = imageIterator.Get();
                        imageIterator.Set(1/n);
                        ++imageIterator;
                    }
                    
                    
                    
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                    multiplyByArtifactMask->SetInput1(myJacobian);
                    multiplyByArtifactMask->SetInput2(inputArtifactMasks[i]);
                    multiplyByArtifactMask->Update();
                    
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldPush[i]);
                    
                    ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                    resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                    resampleI->SetSize( imageSize );
                    resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleI->SetOutputSpacing( imageSpacing );
                    resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleI->SetTransform(dispTransform);
                    resampleI->SetDefaultPixelValue(0);
                    resampleI->Update();
                    
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(weightImageFinal);
                    addImage->SetInput2(resampleI->GetOutput());
                    addImage->Update();
                    weightImageFinal = addImage->GetOutput();
                }
                
                templateImageFinal = DivideImage(templateImageFinal,weightImageFinal);
            }
            
            if (similarityCostType=="SSD")
            {
                for (int i=1; i<numberOfInputs; i++)
                {
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldPush[i]);
                    
                    MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                    multiplyByArtifactMask->SetInput1(originalInputImages[i]);
                    multiplyByArtifactMask->SetInput2(inputArtifactMasks[i]);
                    multiplyByArtifactMask->Update();
                    
                    ResampleImageFilterType::Pointer   resampleI = ResampleImageFilterType::New();
                    resampleI->SetInput(multiplyByArtifactMask->GetOutput());
                    resampleI->SetSize( imageSize );
                    resampleI->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleI->SetOutputSpacing( imageSpacing );
                    resampleI->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleI->SetTransform(dispTransform);
                    resampleI->SetDefaultPixelValue(0);
                    resampleI->Update();
                    
                    /// extra for SSD, compute |D\phit_t_i|
                    JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                    JacobianImage->SetInput(dispFieldPush[i]);
                    JacobianImage->Update();
                    
                    // extra for SSD, multiply by |D\phit_t_i|
                    MultiplyImageType::Pointer         multiplyJacobian =  MultiplyImageType::New();
                    multiplyJacobian->SetInput1(resampleI->GetOutput());
                    multiplyJacobian->SetInput2(JacobianImage->GetOutput());
                    multiplyJacobian->Update();
                    
                    // extra for SSD
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(templateImageFinal);
                    addImage->SetInput2(multiplyJacobian->GetOutput());
                    addImage->Update();
                    
                    templateImageFinal = addImage->GetOutput();
                }
                
                for (int i=1; i<numberOfInputs; i++)
                {
                    // different for SSD
                    DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                    dispTransform->SetDisplacementField(dispFieldPush[i]);
                    
                    //different for SSD
                    ResampleImageFilterType::Pointer   resampleMask = ResampleImageFilterType::New();
                    resampleMask->SetInput(inputArtifactMasks[i]);
                    resampleMask->SetSize( imageSize );
                    resampleMask->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                    resampleMask->SetOutputSpacing( imageSpacing );
                    resampleMask->SetOutputDirection( inputImages[0]->GetDirection() );
                    resampleMask->SetTransform(dispTransform);
                    resampleMask->SetDefaultPixelValue(0);
                    resampleMask->Update();
                    
                    // differenct for SSD
                    JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                    JacobianImage->SetInput(dispFieldPush[i]);
                    JacobianImage->Update();
                    
                    // differenct for SSD
                    MultiplyImageType::Pointer         multiplyByArtifactMask =  MultiplyImageType::New();
                    multiplyByArtifactMask->SetInput1(JacobianImage->GetOutput());
                    multiplyByArtifactMask->SetInput2(resampleMask->GetOutput());
                    multiplyByArtifactMask->Update();
                    
                    // differenct for SSD
                    AddImageType::Pointer    addImage = AddImageType::New();
                    addImage->SetInput1(weightImageFinal);
                    addImage->SetInput2(multiplyByArtifactMask->GetOutput());
                    addImage->Update();
                    weightImageFinal = addImage->GetOutput();
                }
                
                templateImageFinal = DivideImage(templateImageFinal,weightImageFinal);
            }
        }
        else
        {
            templateImageFinal = originalInputImages[0];
        }
    
    
            // write template images to the output directory
            for (int i=0; i<numberOfInputs; i++)
            {
                DisplacementTransformType::Pointer    dispTransform = DisplacementTransformType::New();
                dispTransform->SetDisplacementField(dispFieldPull[i]);
                
                ResampleImageFilterType::Pointer   resampleI0 = ResampleImageFilterType::New();
                resampleI0->SetInput(templateImageFinal);
                resampleI0->SetSize( imageSize);
                resampleI0->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                resampleI0->SetOutputSpacing( imageSpacing );
                resampleI0->SetOutputDirection( inputImages[0]->GetDirection() );
                resampleI0->SetTransform(dispTransform);
                resampleI0->SetDefaultPixelValue(0);
                resampleI0->Update();
                
                //// estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
                /// this implements Equation 5.16 in Wei's thesis
                JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
                JacobianImage->SetInput(dispFieldPush[i]);
                JacobianImage->Update();
                
                DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
                defJac->SetDisplacementField(dispFieldPull[i]);
                
                ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
                resampleJac->SetInput(JacobianImage->GetOutput());
                resampleJac->SetSize( imageSize );
                resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
                resampleJac->SetOutputSpacing( imageSpacing );
                resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
                resampleJac->SetTransform(defJac);
                resampleJac->SetDefaultPixelValue(1);
                resampleJac->Update();
                
                ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
                
                
                itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
                imageIterator.GoToBegin();
                while(!imageIterator.IsAtEnd())
                {
                    // Get the current pixel
                    float n = imageIterator.Get();
                    imageIterator.Set(1/n);
                    ++imageIterator;
                }
                
                MultiplyImageType::Pointer         multiplyByJacobian =  MultiplyImageType::New();
                multiplyByJacobian->SetInput1(resampleI0->GetOutput());
                multiplyByJacobian->SetInput2(myJacobian);
                multiplyByJacobian->Update();
                
                
                ImageWriterType::Pointer ImageWriter = ImageWriterType::New();
                ImageWriter->SetFileName(outputDirectory + "template_image_at_time_" +
                                         std::to_string(inputImageTimes.at(i)) + ".nii.gz");
                ImageWriter->SetInput(multiplyByJacobian->GetOutput());
                ImageWriter->Update();
            }
    
    
    if (outputDispType=="Eulerian")
    {
        cout << "output type = Eulerian" << endl;
        // write displacement field
        for(int i=0; i<numberOfInputs; i++)
        {
           cout << "write eulerian displacement fields" << endl;
            VectorImageWriterType::Pointer  dispFieldWriter = VectorImageWriterType::New();
            dispFieldWriter->SetFileName(outputDirectory + "dispField_pull_at_time_" + std::to_string(inputImageTimes.at(i)) + ".nii.gz");
            dispFieldWriter->SetInput(dispFieldPull[i]);
            dispFieldWriter->Update();
        }
        
        // compute Jacobian Image at each input time points
        for(int i=0; i<numberOfInputs; i++)
        {
            
            //// estimate Jacobian of phi[i]^{-1} by Jacobian of phi[i]
            JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
            JacobianImage->SetInput(dispFieldPush[i]);
            JacobianImage->Update();
            
            DisplacementTransformType::Pointer   defJac = DisplacementTransformType::New();
            defJac->SetDisplacementField(dispFieldPull[i]);
            
            ResampleImageFilterType::Pointer   resampleJac = ResampleImageFilterType::New();
            resampleJac->SetInput(JacobianImage->GetOutput());
            resampleJac->SetSize( imageSize );
            resampleJac->SetOutputOrigin(  inputImages[0]->GetOrigin() );
            resampleJac->SetOutputSpacing( imageSpacing );
            resampleJac->SetOutputDirection( inputImages[0]->GetDirection() );
            resampleJac->SetTransform(defJac);
            resampleJac->SetDefaultPixelValue(1);
            resampleJac->Update();
            
            ImageType::Pointer   myJacobian =  resampleJac->GetOutput();
            
            
            itk::ImageRegionIterator<ImageType> imageIterator(myJacobian,myJacobian->GetBufferedRegion());
            imageIterator.GoToBegin();
            while(!imageIterator.IsAtEnd())
            {
                // Get the current pixel
                float n = imageIterator.Get();
                imageIterator.Set(1/n);
                ++imageIterator;
            }
            
            
            ImageWriterType::Pointer JacobianImageWriter = ImageWriterType::New();
            JacobianImageWriter->SetFileName(outputDirectory + "Jacobian_pull_at_time_" + std::to_string(inputImageTimes.at(i)) + ".nii.gz");
            JacobianImageWriter->SetInput(myJacobian);
            JacobianImageWriter->Update();
        }
    }
    
    if (outputDispType=="Lagrangian")
    {
        //// write displacement field
        for(int i=0; i<numberOfInputs; i++)
        {
            VectorImageWriterType::Pointer  dispFieldWriter = VectorImageWriterType::New();
            dispFieldWriter->SetFileName(outputDirectory + "dispField_push_at_time_" + std::to_string(inputImageTimes.at(i)) + ".nii.gz");
            dispFieldWriter->SetInput(dispFieldPush[i]);
            dispFieldWriter->Update();
        }
        
        //// compute Jacobian Image at each input time points
        for(int i=0; i<numberOfInputs; i++)
        {
            JacobianFilterType::Pointer         JacobianImage = JacobianFilterType::New();
            JacobianImage->SetInput(dispFieldPush[i]);
            JacobianImage->Update();
            ImageWriterType::Pointer JacobianImageWriter = ImageWriterType::New();
            JacobianImageWriter->SetFileName(outputDirectory + "Jacobian_push_at_time_" + std::to_string(inputImageTimes.at(i)) + ".nii.gz");
            JacobianImageWriter->SetInput(JacobianImage->GetOutput());
            JacobianImageWriter->Update();
        }
    }
    
    file.close();
    
    return EXIT_SUCCESS;
}

