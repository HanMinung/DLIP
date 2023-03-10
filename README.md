# Deep Learning Image Processing

* writer : HanMinung
* School of Mechanical and Control Engineering, Handong Global University
* Date : 2023 spring semester
* Image processing
  * software : C++ , Visual Studio 2019 , OpenCV 3.5.13
* Deep Learning with YOLO
  * CUDA
  * cuDNN
  * pyTorch
* Purpose 
  * Information on the fundamental principles of various techniques in classical image processing
  * Object detection using deep learning model : YOLO

---

[TOC]

## Image processing

### 1. CNN

- Convolution Neural Network

* Convolution 연산을 통한 이미지 데이터 filtering

* CNN은 이미지에서의 특징 추출에서 적극적으로 활용된다.

  * Kernel 이라는 필터를 이용하기 떄문

  <img src="https://user-images.githubusercontent.com/99113269/224066554-fbcdd0da-323e-497e-b05b-cb28a84f2e83.png" alt="image" style="zoom:50%;" />

* 커널을 이용해 원본 이미지에 convolution을 취하면 필터의 트성에 맞게 강조된 이미지를 얻을 수 있다.

* Stride

  * 필터링은 필터를 이미지에 놓고 움직이면서 실시하게 된다.

  * 이때, 필터가 한번에 움직이는 거리를 stride라고 한다.

  * 원본 이미지의 크기가 mxm이고, 필터 커널의 크기가 nxn, stride = s 라고하면, 필터링을 통해 얻은 이미지의 크기는  (m-n)/s+1 x (m-n)/s+1 이 된다. 

  * 예시

    <img src="https://user-images.githubusercontent.com/99113269/224068090-2f7d4c22-0fec-4642-b865-2231c8db894b.png" alt="image" style="zoom:50%;" />

* Padding

  * Convolution 연산을 수행하기 전 Input image 주변에 특정 값을 채워 사이즈를 늘리는 과정
  * 이미지의 가장자리 픽셀 정보가 유실되는 것을 방지하기 위한 기능
  * feature map의 크기가 줄어드는 것을 방지
  * 주로, 주변부에 아무런 값이 없는 0을 넣는 zero-padding을 많이 사용

* n x nfilter window : 총 power(n,2)에 해당하는 곱연산을 수행하게 된다.



### 2. Spatial filter

#### 2.1. 2D convolution of filter window

<img src="https://user-images.githubusercontent.com/99113269/224071475-aed154f7-2eb4-4abc-b34c-9160c4baf982.png" alt="image" style="zoom:50%;" />

* General representation : normalization

  <img src="https://user-images.githubusercontent.com/99113269/224071750-44928357-66be-41d3-ab8f-ac92a85d4988.png" alt="image" style="zoom:50%;" />

* dividing factor : total summation of image kernel

  <img src="https://user-images.githubusercontent.com/99113269/224072161-a8916299-2ff9-4c29-9f85-d565e7d46e59.png" alt="image" style="zoom:50%;" />



### 2.2. Commonly used spatial filter mask

#### 2.2.1. Smoothing filter

- average of the pixels in the neighborhood of the filter mask.

- Removal of small details by blurring and reduces sharp intensity of noise

  <img src="https://user-images.githubusercontent.com/99113269/224074476-353d0d3b-f459-4fd2-b0d2-776c69dbedd3.png" alt="image" style="zoom:40%;" />



#### 2.2.2. Sharpening filter

* Method using derivative

  * intensity가 잘보이게 하기 위해, 2차 미분을 사용한다.

  * 2차 미분 중에서도, Laplacian 미분(linear isotropic operator)을 적용한다.

    <img src="https://user-images.githubusercontent.com/99113269/224081842-8292fbfb-1e73-46d4-a86b-d85bda090a5d.png" alt="image" style="zoom:50%;" />

  * Formation of filter

    <img src="https://user-images.githubusercontent.com/99113269/224082832-97a1c675-35c6-4259-a415-282f25755034.png" alt="image" style="zoom:50%;" />

  * To sharpen an image, add Laplacian image to the original image

    <img src="https://user-images.githubusercontent.com/99113269/224082990-26bf17c7-c7bd-4583-9a1f-4881924dcb6a.png" alt="image" style="zoom:50%;" />

  * c : scaling factor

  * Sample code

    ```c++
    int main() {
    	Mat image, laplacian, abs_laplacian, sharpening;
    	image = imread("path/Moon.png", 0);
    
    	// calculates the Laplacian of an image
    	// - image: src, laplacian: dst, CV_16S: desire depth of dst,
    	// - 1: aperture size used to compute second-derivative (optional)
    	// - 1: optional scale factor for the computed Laplacian values
    	// - 0: optional delta value that is added to the result
    	Laplacian(image, laplacian, CV_16S, 1, 1, 0);
    	convertScaleAbs(laplacian, abs_laplacian);
    	sharpening = abs_laplacian + image;
    	
        imshow("Input image", image);
    	imshow("Laplacian", laplacian);
    	imshow("abs_Laplacian", abs_laplacian);
    	imshow("Sharpening", sharpening);
    	waitKey(0);
    }
    ```

  

* Unsharp masking

  * original image에서 blurred image(averaged image)를 빼면 shrap한 부분만 검출 할 수 있다.

  * unsharp mask = original signal - blurred signal

  * Sample code

    ```c++
    int main() {
        
    	Mat image, avg_image, unsharp_mask, sharpening;
    	image = imread("path/Moon.png", 0);
    	unsharp_mask.convertTo(unsharp_mask, CV_16S);
    	sharpening.convertTo(unsharp_mask, CV_16S);
    
    	blur(image, avg_image, Size(9, 9));
    	unsharp_mask = image - avg_image;
    	sharpening = image + (0.5 * unsharp_mask);
    
    	imshow("Input image", image);
    	imshow("Sharpening", sharpening);
    
    	waitKey(0);
    }
    ```

    



### 2.3. Filter implementation in C++

* Comparison of effect of each filters

  * Original image

    <img src="https://user-images.githubusercontent.com/99113269/224306172-8c21e8f3-5166-44e8-9dbf-36e9b9980499.png" alt="image" style="zoom:50%;" />

  * Blur, Gaussian blur, Median blur

    ```c++
    size : kernel size
    
    /* Blur */
    cv::blur(src, dst, cv::Size(i, i), cv::Point(-1, -1));
    
    /* Gaussian Filter */
    cv::GaussianBlur(src, dst, cv::Size(i, i), 0);
    
    /* Median Filter */
    cv::medianBlur(src, dst, 3);
    ```

    * Result

      <img src="https://user-images.githubusercontent.com/99113269/224307615-c20dd1e4-5506-4eb8-a89a-89c718a095db.png" alt="image" style="zoom: 67%;" />

  * Laplacian filter & Result

    ```c++
    /* Laplacian Filter */
    int kernel_size = 3;
    int scale		= 1;
    int delta		= 0;
    int ddepth		= CV_16S;
    
    cv::Laplacian(src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
    src.convertTo(src, CV_16S);
    cv::Mat result_laplcaian = src - dst;
    result_laplcaian.convertTo(result_laplcaian, CV_8U);
    
    namedWindow("Laplacian", CV_WINDOW_AUTOSIZE);
    cv::imshow("Laplacian", result_laplcaian);
    ```

    <img src="https://user-images.githubusercontent.com/99113269/224308231-f8cdbd42-5ddd-4d74-9095-7e571da68818.png" alt="image" style="zoom:45%;" />

  * 

    

    

    

    

    

    

    

    

    
