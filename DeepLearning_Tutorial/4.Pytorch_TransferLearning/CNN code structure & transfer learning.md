# Code structure of CNN

Writer : HanMinung

Date : Spring semester, 2023

Purpose : Understanding the code structure of deep neural network 

--------------

## Inference using pre-trained model (classification) - part 3-1

```python
from torchvision import models

dir(models)
```

* Torchvision
  * offers a collection of pre-trained models, including popular architectures like VGG, ResNet, and AlexNet.
* dir(models)
  * list of all the names (attributes) defined in the `models` module within the torchvision library.
  * Model includes 'ALEXNET', 'GOOGLENET', 'RESNET'



```python
model = models.vgg16(pretrained = True)
model.eval()  
```

* Use VGG16 model
* evaluation mode
  * 학습과정에서 사용되는 드롭아웃(Dropout)이나 배치 정규화(Batch Normalization)과 같은 층들이 평가 시에 일관된 동작을 수행
  * Predefined model을 사용하기 때문에, evaluation mode를 사용
  * Dropout : 모델의 일부 유닛을 무작위로 비활성화하여 과적합을 줄이는 정규화 기법
  * Batch normalization : 입력 데이터를 정규화하여 학습을 안정화시키고 속도를 향상시키는 방법



```python
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

* **transforms.Resize(256)** : 이미지의 크기를 조정
* **transforms.CenterCrop(224)** : 이미지의 중앙 부분을 잘라내어 크기를 224x224로 만든다
* **transforms.ToTensor()** : 이미지를 PyTorch의 텐서로 변환. 이미지를 숫자로 표현하는 텐서 형태로 변환하여 딥러닝 모델에 입력으로 제공하기 위한 작업입
* **transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])** : 이미지의 채널별 픽셀 값을 정규화. 입력 이미지의 RGB 채널 값들을 평균(mean)과 표준편차(std)를 사용하여 정규화



```python
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)

input_tensor = preprocess(input_image)  # Resize image (pre processing)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# meaning of torch.no_grad()
# pytorch : auto grad engine이라는 것이 있다.
# auto grad engine : training code를 돌릴 때, gradient 값을 기억하고, 역전파 연산을 진행할 수 있도록 만들어두었음
# torch.no_grad : 사용되는 메모리를 줄일 수 있고, 연산량을 줄일 수 있다. 

# 전체적인 flow for lighter model : torch.eval --> torch.no_grad
# 학습을 시키고자 한다면 : eval 모드가 아닌, model.train

with torch.no_grad():
    output = model(input_batch)
    
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# softmax : change output to zero to one (probability check)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

* PIL : python image library

* preprocess (전처리 파이프라인)을 사용하여 입력 이미지를 전처리. (사이즈 조정, 중앙 자르기, 텐서 변환 및 정규화  등등의 과정)

* 모델이 기대하는 형태의 입력 미니배치를 생성하기 위해 `input_tensor`에 차원을 추가

* Meaning of torch grad

  * 현재는, pretrained 모델을 사용하기 때문에, no_grad로 설정하나, 학습모드로 설정하면 `model.train`으로 설정

  * 컨텍스트 내에서 실행되는 부분은 gradient 연산 & back propagation을 수행하지 x
  * 메모리 사용량과 연산량을 줄일 수 있다.
  * input batch를 사용하여 예측을 수행

* softmax를 적용하여 0 ~ 1 사이의 확률값으로 변환





## Transfer Learning using Pre-trained Models - part 3-2

* Transfer learning
  * ImageNet과 같은 대량의 데이터셋으로 이미 학습이 되어있는 모델을 사용
  * 기존에 생성되어 있는 모델들을 가지고 다른 데이터셋에 적용시켜보는 것
  * Computer vision 분야에서 다양하게 사용되는 방법
  * 학습된 모델을 기반으로 최종 출력층을 바꿔 학습하는 기법
  * Workflow
    * 학습된 모델의 최종 출력층을 보유 중인 데이터에 대응하는 출력층으로 바꾼다.
    * 교체한 출력층의 결합 파라미터를 소량의 데이터로 다시 학습
    * 입력층에 가까운 부분의 결합 파라미터를 변화시키지 않음
* Fine tuning
  * 출력층 등을 변경한 모델을 학습된 모델을 기반으로 구축
  * 직접 준비한 데이터로 신경망 모델의 결합 파라미터 학습
  * 전이 학습과 달리, 모든 층의 파라미터 재학습

```python
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception*]
model_name = "resnet"

# Number of classes in the dataset (ant & bee)
# In transfer learning, we can change the outptu classes
num_classes = 2

'''
    True (feature extraction) : only update the reshaped layer params,
    False(finetuning)         : finetune the whole model, 
'''

feature_extract = True  

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

from torchsummary import summary

summary(model_ft, (3,input_size,input_size))

print(model_ft)
```

* Resnet이라는 pretrained model을 사용
* numclasses : 분류하고자 하는 class의 수
* `feature_extract = True ` : 전이 학습 or Fine tuning을 결정하는 부분
  * Feature extraction : 사전에 훈련된 모델의 마지막 층만 재조정하고, 나머지는 고정
* initialize_model에서 

```python
# Top level data directory. Here we assume the format of the directory conforms 
# to the ImageFolder structure
data_dir = "./hymenoptera_data"

# Data augmentation and normalization for training
# Just normalization for validation
# Normalized with ImageNet mean and variance
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

training_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform['train'])
test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform['val'])

classes = ['ant', 'bee']
print(f"train dataset length = {len(training_data)}")
print(f"test  dataset length = {len(test_data)}")
```

* 훈련 데이터 및 검증 데이터에 대한 데이터 증강 및 정규화 작업을 진행
* 데이터 전처리 : 모델 학습에 필요한 데이터를 적절한 형태로 만드는 과정
  * `RandomResizedCrop(input_size)`: 이미지를 무작위로 잘라서 일정한 크기 (input_size)로 만드는 방법, 모델이 과적합되는 것을 방지하고 다양한 스케일과 비율을 가진 이미지에 대한 모델의 견고성을 향상
  * `RandomHorizontalFlip`: 이미지를수평으로 무작위로 뒤집는 방식, 데이터 증강의 한 형태
  * `ToTensor`: 이미지 데이터를 PyTorch의 텐서(Tensor) 형태로 변환, 이미지 데이터는 일반적으로 픽셀 강도 값을 0-255 범위의 정수로 가지는데, `ToTensor()`는 이를 0-1 범위의 부동 소수점으로 변환하고, 이미지의 차원을 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 변경합니다. 이는 PyTorch에서 컨볼루션 연산을 수행하는 방식과 일치
* `torchvision.datasets.ImageFolder`는 PyTorch에서 제공하는 이미지 데이터 로딩에 유용한 클래스입니다. 이 클래스는 주어진 디렉토리에서 이미지를 불러오며, 하위 디렉토리 이름을 기반으로 레이블을 자동으로 생성
  * 그 후에, train은 transform중 train에 해당하는 작업을, test set은 transform중 test에 해당하는 작업을 수행
* 전체적인 Flow 정리
  * 기존에 존재하는 "**RESNET**" 모델을 불러옴 (pretrained model)
  * ant, bee 데이터를 가져와서 (train, test) 전이 학습을 수행 
  * feature_extract = True로 설정하면, 출력층에만 변화를 주는데 이 역할은 `initialize_model`함수에서 수행
  * feature_extract = False로 설정하면, fine tuning을 수행하며, 역전파를 통해 모든 파라미터를 다시 찾게됨

