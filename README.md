# YOLOv3 pytorch implementation in python 2.7
## Table of contents
- [Environment](#Environment)
- [Installation](#Installation)
- [Custom Dataset settings](#How-to-construct-your-custom-dataset)
- [Training YOLO model](#How-to-train-YOLOv3-model-with-your-custom-dataset)
- [Testing weights](#How-to-test-YOLOv3-model-weights)
- [Detecting objects](#How-to-detect-objects-in-images-with-trained-YOLO-model)

## Environment
> - Ubuntu 18.04, Python 2.7
> ### Before you use this repository, we recommend you to install CUDA toolkit and cudnn
> - Python 2.7 and torch 1.4.0, the recommended compatible version of CUDA toolkit is  10.x (10.0, 10.1 or 10.2) and cudnn is 7.6.5

## Installation
> ### Installing from github
> ```bash
> git clone https://github.com/hjinnkim/yolov3-python2.7.git
> cd yolov3-python2.7/
> python -m pip install -r requirements.txt
> ```
> Above commands let you download this repository and install the required python packages
> ### Download pretrained weights
> ```bash
> sh ./weights/download_weights.sh
> ```
> ### If you finished following above instructions, your directory looks like below:
> ![Screenshot from 2022-04-26 09-30-07](https://user-images.githubusercontent.com/89929547/165195729-c44aa8a7-4eef-4500-a7d2-62731d48a7a8.png)


## How to construct your custom dataset
Actually, you can place your custom dataset anywhere in your workstation, but we recommend that you follow below method.
> ### 1. Place your custom dataset under *"data"* directory
> ![Screenshot from 2022-03-22 11-18-38](https://user-images.githubusercontent.com/89929547/159394789-c02226ea-c7fb-4515-be42-e8df4909766a.png)
> ### You can see the overall custom dataset structure. We will explain the structure one by one.
> 1.1. In your custom dataset directory, you have two directories, *"images"*, *"labels"*, and two text files, *"train.txt"*, *"val"*.
> 
> 1.2. In the *"train"* directory inside the *"images"* directory, there are images for training your model. In the *"val"* directory inside the *"images"* directory, there are images for validating your model. 
> 
> 1.3. In the *"train"* directory inside the *"labels"* directory, there are labels for training images (ground truth). In the *"val"* directory inside the *"images"* directory, there are labels for validating images.
> 
> 1.4. The corresponding image and label must have same name except for file format. 
> ![Screenshot from 2022-03-22 11-48-43](https://user-images.githubusercontent.com/89929547/159398074-82db19a7-26d8-4193-8ef8-99afa8588208.png)
> 
> Above structure is an example. Your custom dataset can have as many images as you want.
> 
> 1.5. *"train.txt"* contains training image absolute paths. *"valid.txt"* contains validating image absolute paths. 
> ~~~
> # In your train.txt file
> /home/[username]/yolov3_python2.7/data/custom/images/train/0196.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/train/210805_0307.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/train/0463.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/train/210805_0214.jpg
> ...
> ~~~
> ~~~
> # In your valid.txt file
> /home/[username]/yolov3_python2.7/data/custom/images/val/v_0519.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/val/v_0464.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/val/v_0476.jpg
> /home/[username]/yolov3_python2.7/data/custom/images/val/v_0275.jpg
> ...
> ~~~
> 
> ### 2. Make **"custom.names"** file in **"data"** directory
> **"custom.names"** file contains class names of your custom dataset
> 
> ![Screenshot from 2022-03-22 13-23-25](https://user-images.githubusercontent.com/89929547/159407697-b862a34b-742c-427d-a014-7089dcf0f893.png)
> 
> ### 3. Make **"custom.data"** file in *"config"* directory
> **"custom.data"** file contains the number of classes and relative paths of **"custom.names"** file, **"train.txt"** file, **"valid.txt"** file. The base directory is *"yolov3-python2.7"*.
> 
> ![Screenshot from 2022-03-22 13-22-55](https://user-images.githubusercontent.com/89929547/159407729-1c5be8b6-0483-4b84-8b91-09181f190f8a.png)

## How to train YOLOv3 model with your custom dataset
> ### YOLOv3 model structure is described in cfg file. The cfg file are recommended to be contained in **config"" directory.
> ~~~
> config/yolov3.cfg
> config/yolov3-tiny.cfg
> ~~~
> 
> Using the cfg file, you can build your YOLO model.
> ### You can train YOLOv3 model with your custom dataset by running the below command:
> ```bash
> # Make sure that your current working directory is yolov3-python2.7
> python yolo/train.py
> ``` 
>
> In this case, you train YOLO model with some default settings. The default settings are as follows:
> ~~~
> 1. Using "YOLOv3-tiny" model
> 2. Using pretrained *"YOLOv3-tiny"* weights
> 3. Training epochs : 100
> 4. Using dataset information in "config/custom.data"
> and some other parameters
> ~~~
> These default settings can be checked in the run() function in **"train.py"** code. If you want to change traing settings, you can directly change default settings in **"train.py"** code.
> 
> ### Also, you can give arguments when running **"train.py"** in command line
> ```bash
> # For example
> python yolo/train.py --model "config/yolov3.cfg" --pretrained_weights "weights/yolov3.weights" --epochs 30
> ``` 
> #### 1. You can choose model with option *"--model"*
> ```bash
> # If you want to use YOLOv3 model
> python yolo/train.py --model "config/yolov3.cfg" 
> ```
> #### 2. You can choose the pretrained weights with option *"--pretrained_weights"*
> ```bash
> # If you want to use the pretrained YOLOv3 weights
> python yolo/train.py --pretrained_weights "weights/yolov3.weights"
> ```
> 
> When choosing model structure and the pretrained weights, **you must choose the corresponding weights with the model.** For example, if you want to use YOLOv3 model, you need to use the pretrained weights trained in YOLOv3. If not, error will be occur.
>
> #### 3. You can change training epochs with option *"--epochs"*
> ```bash
> # If you want to train the model by 30 epochs
> python yolo/train.py --epochs 30
> ```
> #### 4. You can change dataset information with option *"--data"*
> ```bash
> # If you want to change the dataset information
> python yolo/train.py --data "config/custom.data"
> ```
> 
> The dataloader for the model is based on the path inside the **.data** file. You have to exactly write the path inside the **.data** file.
> 
> ### Other options
> - **n_cpu**
>   This option is for pytorch dataloader. The defulat value is 8. We recommend you to check the number of cpu cores by running follow command:
>   ```bash
>   lscpu | grep Core
>   ```
>   Then, the following output will be showing
>   
>   ![Screenshot from 2022-03-22 14-39-20](https://user-images.githubusercontent.com/89929547/159415234-bcf1d6b5-583b-4ce1-848f-6f2cdc3a2743.png)
>   
>   We recommend you to set **n_cpu** as the number of cores
>   ```bash
>   python yolo/train.py --n_cpu 8
>   ```
> - **verbose**
>     This options will show you more detailed training result at each epoch.
>   ```bash
>   python yolo/train.py --verbose
>   ```
> ### Trained weights are saved in "checkpoints" directory
> ```bash
> checkpoints/custom_weight_{epoch}.pth
> ``` 
> ### Using GPU accelearation
> If your workstation has the nvidia gpu and you have installed cuda toolkit, the gpu will be used during training. You can check gpu memory usage and gpu usage by following command:
> ```bash
>  watch -n 0.5 nvidia-smi
>  ```
> If **CUDA out of memory** error occurs, you need to change the parameter inside **.cfg** file. In the **.cfg** file, you can see *batch* parameter.
> 
> ![Screenshot from 2022-03-22 15-07-33](https://user-images.githubusercontent.com/89929547/159418387-bfa285a4-74d2-4406-9eeb-df958e3c0e1f.png)
> 
> You have to decrease the *batch* parameter until **CUDA out of memory** error doesn't occurs.
>  
> In opposite case, if gpu memory usage is too low, I recommend you to increase the parameter. 
> 
> We recommend you set the parameter as power of 2.
## How to test YOLOv3 model weights
> ### It is very similar to training
>  ```bash
>  # For example
>  python yolo/test.py --weights "checkpoints/custom_weight_{epochs}.pth" --data "config/custom.data" 
>  ```
> Then, the positive true mAP will be shown
## How to detect objects in images with trained YOLO model
> You can detect obejcts in images with your custom trained YOLO model
> <br>1. Make a directory for detected images. Place images in the directory.
> <br>2. Give an option for the detected directory path when running **detect.py** code. (default option is **data/samples**)
> <br>3. Give an option for model configuration file path when running **detect.py** code. (default option is **config/yolov3-tiny.cfg**)
> <br>4. Give an option for the trained weights (default option is **weights/yolov3-tiny.weights**)
> <br>5. Give an option for the class name (default option is **data/custom.names**)
>  ```bash
>  # For example
>  python yolo/detect.py --images "data/samples" --weights "checkpoints/custom_weight_{epochs}.pth" --classes "data/custom.names" 
>  ```
>  Then, the output images will be generated in **data/output** (**data/output** directory is default option. If the given option path is not directory, the running code will create the directory automatically.)

## Reference
The reference for this repository is https://github.com/eriklindernoren/PyTorch-YOLOv3
