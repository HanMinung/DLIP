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



#### 2.2. Commonly used spatial filter mask

- average of the pixels in the neighborhood of the filter mask.

- Removal of small details by blurring and reduces sharp intensity of noise

  <img src="https://user-images.githubusercontent.com/99113269/224074476-353d0d3b-f459-4fd2-b0d2-776c69dbedd3.png" alt="image" style="zoom:40%;" />



##### 2.2.1. Sharpening filter

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

    



#### 2.3. Filter implementation in C++

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



### 3. Thresholding& Morphology

----

#### 3.1. T(r)  :  s(i, j) = T(r( i, j ))

* Changes the intensity of individual pixels, from level r (input) to s (output)

* Histogram : Probability of the occurrence of intensity level r_k

  <img src="https://user-images.githubusercontent.com/99113269/224997865-f13c5a02-5ddf-4a02-8790-c66c5affb0ce.png" alt="image" style="zoom: 50%;" />



#### 3.2. Histogram equalization

##### 3.2.1. Definition

* For object segmentation, it is preferable to have a high contrast image

  <img src="https://user-images.githubusercontent.com/99113269/224998430-9bedeb09-cb94-4b61-8413-9fa155339147.png" alt="image" style="zoom: 38%;" />

##### 3.2.2. Local Histogram equalization

* Divide image into sections and apply histogram statistics locally for image enhancement

  <img src="https://user-images.githubusercontent.com/99113269/224999116-59c610c6-926e-4eac-8c0b-c0cec180bc4c.png" alt="image" style="zoom:38%;" />

* Global thresholdong : Basic global thresholding / Otsu's method

  Local thresholding : Variable threshold values for each sub-window



##### 3.2.3. Global Binary Thresholding

* Different methods for thresholding (openCV functions)

<img src="https://user-images.githubusercontent.com/99113269/225000133-175c070d-69a4-4605-b5b6-1a761dd4b395.png" alt="image" style="zoom:40%;" />

* Apply when the object and background has distinct intensity distribution

* Minimize the average error occurred om segmented groups

* An iterative algorithm of finding T

  - Initial estimation of T ( usually mean of image intensity )

  - segment the image using T

  - find mean of Ga and G2

  - compute new T value

  - repetition

    <img src="https://user-images.githubusercontent.com/99113269/225001124-e515d766-0fd4-41d7-8e4a-17561807aae4.png" alt="image" style="zoom:50%;" />





##### 3.2.4. Optimum global threshold - Otsu method

* Optimum threshold by maximizing the between-class variance

* Algorithm for finding critical point without any repetition

* 임계값을 임의로 정해 픽셀을 두 부류로 나누고 두 부류의 명암 분포를 구하는 작업을 반복

* 모든 경우의 수 중에서 두 뷰류의 명암 분포가 가장 균일할 때의 임계값을 선택한다.

* Procedure

  * 히스토그램에서 가중치 분산을 최소화한 결과로 주어진 임계값 t를 사용하여 구 클래스로 분리한다. 

  * Within class variance

    <img src="https://user-images.githubusercontent.com/99113269/225005401-3e4b7ede-6553-4363-b865-7ae899295a7f.png" alt="image" style="zoom:50%;" />

  * Within class variance가 최소가 되는 t값을 찾는 것이다. 

  * 따라서, 다음 Between-class variance라는 개념을 새로 정의한다.

    <img src="https://user-images.githubusercontent.com/99113269/225006411-df3a7e0d-5340-4a17-8690-045192a31906.png" alt="image" style="zoom: 33%;" />

    <img src="https://user-images.githubusercontent.com/99113269/225007832-b69deabd-e91c-46e3-b43d-8b36f4dc1ef1.png" alt="image" style="zoom: 50%;" />





##### 3.2.5. Local thresholding

- Apply thresholding differently on segment of images

  <img src="https://user-images.githubusercontent.com/99113269/225009152-b65a24f2-60a2-48f7-b6fd-a79d856b7179.png" alt="image" style="zoom:40%;" />







#### 3.3. Morphology

* After the threshold, small pieces of broken segments are seen
* There exists two methods : Dilation, Erosion



##### 3.3.1. Dilation

* Definition : LOGIC OR
* 하나라도 겹치면 1로 처리
* Effect : thickens / Grows objects, used for bridging gap

<img src="C:\Users\hanmu\AppData\Roaming\Typora\typora-user-images\image-20230314222417841.png" alt="image-20230314222417841" style="zoom:50%;" />



##### 3.3.2. Erosion

* Definition : AND logic

* 모두 겹쳐야 1로 처리

* Erosion example

  <img src="https://user-images.githubusercontent.com/99113269/225014804-ddecad21-9ff0-43ed-b652-329251e622b1.png" alt="image" style="zoom:50%;" />



##### 3.3.3. Opening and Closing

* Opening

  * To smooth the contour of an object, eliminate thin protrusions.

  * erosion of A by B  --> dilate by B

  * Erosion을 하여 잔 점들을 모두 없애고, 후에 dilate를 진행

    <img src="https://user-images.githubusercontent.com/99113269/225015620-b7cdc878-7b70-4ae5-8be5-e2cdee1305e7.png" alt="image" style="zoom: 67%;" />

    

* Closing

  * Dilate A by B --> Erode by B

  * Effect ( refer below example )

    <img src="https://user-images.githubusercontent.com/99113269/225015930-ae0c9ad1-fc5c-4792-bd5f-b0823acc62a9.png" alt="image" style="zoom: 42%;" />







 #### 3.4. Sample  code

##### 3.4.1. Threshold function

```c++
[threshhold - type]
	0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
        
- Function implementation
threshold(src, dst, threshold_value, max_binary_value, threshold_type);
```



##### 3.4.2. Track bar

* Trackbar의 경우, interrupt와 같이 변화가 있을때 마다 demo 함수가 실행이 된다.

```c++
// [Trackbar variable]
int threshold_value = 0;
int threshold_type	= 0;
int Cval = 0;
int morphology_type = 0;

int const max_value = 255;
int const max_type	= 6;
int const max_C = 10;
int const max_binary_value = 255;

// [Trackbar strings]
String window_name		= "Threshold & Morphology Demo";
String trackbar_type	= "Thresh Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero 								   Invertd \n 5: Otsu method";
String trackbar_value	= "Thresh Value";
String trackbar_morph	= "Morph Type 0: None \n 1: erode \n 2: dilate \n 3:  \n 4: open";

// Create trackbar to choose type of threshold
createTrackbar(trackbar_type,	window_name, &threshold_type,	max_type,	Threshold_Demo);
createTrackbar(trackbar_value,	window_name, &threshold_value,	max_value,	Threshold_Demo);
createTrackbar(Cvalue, window_name, &Cval, max_C, Threshold_Demo);
createTrackbar(trackbar_morph,	window_name, &morphology_type,	max_type,	Morphology_Demo);

// Call the function to initialize
Threshold_Demo(0, 0);
Morphology_Demo(0, 0);
```



### 4. Feature detection

​			Edge has rapid change in image intensity. Intensity gradient is 2D vector. (Gradient strength, Gradient direction). Gradient direction is perpendicular to edge difrection. Actual edges are ramp-like instead of step edge. Since edge detection uses gradient method (derivative), data should be proceesed like smoothing. 

#### 4.1. Preprocess

* To reduce the noise effect, use smooth filter before edge detection : average filter, median filter, Gaussian filter

* But smoothing blurs edges

  <img src="https://user-images.githubusercontent.com/99113269/230282687-0dc15462-2c65-4640-bfb7-f4d873e6323a.png" alt="image" style="zoom: 67%;" />

<img src="https://user-images.githubusercontent.com/99113269/230282907-c2b9214d-ee91-46f6-ba5c-b89e3b10119d.png" alt="image" style="zoom: 67%;" />

<img src="https://user-images.githubusercontent.com/99113269/230283192-37491333-5ba2-4696-84f9-4872885ac125.png" alt="image" style="zoom:67%;" />



#### 4.2. Canny edge detection

* What is good edge detector?

  * Low error rate : all edges should be found
  * Well localized edge points

* Canny algorithm uses 1st derivative for detection

  

##### 4.2.1. Factors in canny edge detection

* Sigma : size of gaussian filter
* Good values to start with are between 0.6 to 2.4
  * Smaller filters cause less blurring, and allow detection of small, sharp lines.
  * A larger filter increases processing time and causes more blurring.



##### 4.2.2. Non-maxima suppression

​			Check if pixel is local maximum along gradient direction angle. Figure below shows the result of applying nonmaxima suppression.

<img src="https://user-images.githubusercontent.com/99113269/230284934-bec92852-e4c2-4ace-9abc-b879ef5c8bf6.png" alt="image" style="zoom:50%;" />



##### 4.2.3. Thresholding

* Too high thresholding : actual validated edge points will be eliminated.
* Too low thresholding : false edges will be detected



#### 4.3. Line detection

​			Consider parameter space of mc-plane. 
$$
y = mx + c --> c = -xm + y
$$

* A line in the image corresponds to a point in Hough space. 

* A point in the image corresponds to a line in Hough space.

Two different points on a same line in xy plane ( y = mx + c ) intersects in same point in hough space. 

<img src="https://user-images.githubusercontent.com/99113269/230286871-7f7b9b28-5e0c-4233-8bcb-16983e806108.png" alt="image" style="zoom:50%;" />





### Tips about image processing

#### Tip 1 : Showing multiple images in one window

```c++
#include <opencv2/opencv.hpp>
using namespace cv;

void main(){
    
    Mat image1 = imread("image1.jpg");
    Mat image2 = imread("image2.jpg");
    Mat image3 = imread("image3.jpg");
    Mat image4 = imread("image4.jpg");

    std::vector<Mat> images;
    images.push_back(image1);
    images.push_back(image2);
    images.push_back(image3);
    images.push_back(image4);

    Mat result;
    hconcat(images, result);

    imshow("Result", result);
    waitKey(0);

}

```



#### Tip 2 : Trackbar (with example)

```c++
// Set initial value and the max values of variables 
int threshold_value = 0;
int threshold_type	= 0;
int Cval = 0;
int morphology_type = 0;

int const max_value = 255;
int const max_type	= 6;
int const max_C = 10;
int const max_binary_value = 255;

// Trackbar strings
String window_name		= "Threshold & Morphology Demo";
String trackbar_type    = "Thresh Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Invertd \n 5: Otsu method \n 6: Adaptive threshold";
String Cvalue = "C in adaptive filter";
String trackbar_value	= "Thresh Value";
String trackbar_morph	= "Morph Type 0: None \n 1: erode \n 2: dilate \n 3:  \n 4: open";

// main statement ------------------------------------------------------------------------------------------------------------
namedWindow(window_name, WINDOW_NORMAL);

// Create trackbar to choose type of threshold
createTrackbar(trackbar_type,	window_name, &threshold_type,	max_type,	Threshold_Demo);
createTrackbar(trackbar_value,	window_name, &threshold_value,	max_value,	Threshold_Demo);
createTrackbar(Cvalue, window_name, &Cval, max_C, Threshold_Demo);
createTrackbar(trackbar_morph,	window_name, &morphology_type,	max_type,	Morphology_Demo);

// Function headers
void Threshold_Demo	(int, void*);
void Morphology_Demo(int, void*);

// Function definition
void Threshold_Demo(int, void*){

	//	0: Binary	1: Threshold Truncated	  2: Threshold to Zero	3: Threshold to Zero Inverted
	//	4: To zero inverted		5: Otsu method		6: adaptive threshold
	
	if (threshold_type == 5)  threshold_type = 8;
	threshold(src_gray, dst, threshold_value, max_binary_value, threshold_type);

	if (threshold_type == 6)  adaptiveThreshold(src_gray, dst, max_binary_value, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, Cval);
	imshow(window_name, dst);
}

// default form of callback function for trackbar
void Morphology_Demo(int, void*){  

	// 0: None	1: Erode	2: Dilate	3: Close	4: Open 
	
	switch (morphology_type) {

		case 0: dst.copyTo(dst_morph);									break;
		case 1: erode(dst, dst_morph, element);							break;
		case 2: dilate(dst, dst_morph, element);						break;
		case 3: morphologyEx(dst, dst_morph, CV_MOP_OPEN, element);		break;
		case 4: morphologyEx(dst, dst_morph, CV_MOP_CLOSE, element);	break;
	}

	imshow(window_name, dst_morph);
}

```



## .

## .

## .





## Deep Learning

### 1. Activation function

입력을 받아서 활성화 또는 비활성화를 결정하는 데에 사용되는 함수. 



#### 1.1. Sigmoid function



* 음수 값을 0에 가깝게 표현하기 때문에, 입력 값이 최종 레이어에서 미치는 영향이 적어진다 : Vanishing Gradient Problem
* Back-propagation을 계산하는 과정에서는, 활성화 함수의 미분값을 곱하는 과정이 포함되는데, 이 함수의 경우 은닉층의 깊이가 깊다면 오차율을 계산하기 어렵다는 문제가 발생한다. 
* 또한, 함수의 중심이 0이 아니기 때문에, 학습이 느려질 수 있다.



#### 1.2. Tanh function

![image](https://user-images.githubusercontent.com/99113269/235345910-6f671566-f4da-4ad9-a1ed-5da1e8c634e9.png)

* Hyperbolic tangent function

* 입력값이 작아질수록 출력값은 0에 가까워지고, 입력값이 커질수록 출력값은 1에 가까워진다.

* 입력값이 작아질수록/커질수록 기울기(gradient)는 0에 가까워진다.

* 이 역시, vanishing gradient problem이 발생

  

#### 1.3. ReLU function (Rectified Linear Unit function)

![image](https://user-images.githubusercontent.com/99113269/235345923-26e7c87d-b8d2-4cba-b165-b41fd8b05436.png)

* 앞선 두 activate function이 가지는 gradient vanishing 문제를 해결하기 위한 함수
* Most commonly used in CNN



### 2. Deep Neural network

#### 2.1. Notation

![image](https://user-images.githubusercontent.com/99113269/235345942-04f08824-e193-4b35-866a-2a8283f399d7.png)

![image](https://user-images.githubusercontent.com/99113269/235345957-b92c6bb7-a2da-4c1b-a36b-8801221cd41c.png)



#### 2.2. Back propagation method

![image](https://user-images.githubusercontent.com/99113269/235345986-28de18b6-d3be-473c-9b42-2c17ce570749.png)

![image](https://user-images.githubusercontent.com/99113269/235345994-1463959e-c8a4-47be-9c0b-14231d4cb89d.png)

![image](https://user-images.githubusercontent.com/99113269/235346004-e07cadb0-1701-4afa-ad13-c90de109b6d9.png)



### 3. Convolution neural network

<img src="https://user-images.githubusercontent.com/99113269/236402005-fc04a937-bb43-49ff-bc30-92362a64f168.png" alt="image" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/99113269/236402250-458cdb2c-2b5d-463b-bb01-0a6f121a63fe.png" alt="image" style="zoom:50%;" />



### 4. Inception1 (GOOGLENET)

<img src="https://github.com/HanMinung/EmbeddedController/assets/99113269/d5fb473c-6a7e-4004-be76-6484652437c9" alt="image" style="zoom: 50%;" />

<img src="https://github.com/HanMinung/EmbeddedController/assets/99113269/781ddba2-f3c2-414d-957d-97dccafb7283" alt="image" style="zoom:50%;" />

### 5. RESNET (Residual neural network)

* Previous CNN structures (e.g. ALEXNET, VGGNET) has vanishing gradient  problem
* With structure of residual block & shourtcut connection algorithm, the problem was solved.
* Deeper neural network with 1001 layers does not have any vanishing gradient problem with this algorithm since gradient cannot be under '1' with definition of chain rule.

<img src="https://github.com/HanMinung/EmbeddedController/assets/99113269/413c54fa-536a-4e46-8e54-df4228a169b8" alt="image" style="zoom:50%;" />

