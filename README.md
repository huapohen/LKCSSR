# Requirements
torch 1.7 and above for torch.cuda.amp,
cuda 11.1 and above for 3090,
and this project use torch 1.10.0+cu113 torchvision 0.11.1+cu113

numpy

skimage

timm

einops

# Train
## 1. Prepare training data 
The training dataset is made in the same way as <a href="https://github.com/YingqianWang/iPASSR">iPASSR</a>. Please refer to their homepage  and download the dataset as their guidance.

Extract the dataset and put them into the ./data/patches_x4/

look like this:

./data/patches_x4/028083 

./data/patches_x4/028083/hr0.png

./data/patches_x4/028083/hr1.png

./data/patches_x4/028083/lr0.png

./data/patches_x4/028083/lr1.png

## 2. Begin to train
You should specify the number of GPU in train.py.
You should specify the batch_size in train.py, 
and the batch_size look like this: 

batch_size = num_gpus * batch_per_gpu

batch_size = 6 * 16

I used number of 6 GPU-3090 to train the model.

python train.py


# Test
## 1. Prepare test data 
Extract the dataset and put them into the ./data/test/

look like this:

./data/test/LR_x4/

./data/test/LR_x4/0000_L.png

./data/test/LR_x4/0000_R.png



## 2. Begin to test
python test_ssr.py

I used number of 1 GPU-A40 to inference the model.


# Checkpoints:
 The checkpoint are located under the directory of ./checkpoints
 
./checkpoints/SSR_4xSR_epoch57.pth.tar


Model parameters: 0.23 MB

Fetch link: https://pan.baidu.com/s/1Jyk5rQpLTNbwi0IVzirqYg?pwd=qwer 

Fetch code: qwer 

The result images are placed in the link.