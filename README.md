# Yolov8 Aim Assist with IFF

## 免责声明
- 该项目无意用于游戏，也不应该使用。使用风险自负。
- 该项目将不会定期维护，因为它目前已经达到了目标。一旦有更好的模型/压缩方法/量化方法可用，可能会提供更新。
- 该项目是基于[Chalkeys/Yolov8-Apex-Aim-assist-with-IFF](https://github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF)项目的翻译和改进，原项目[Franklin-Zhang0/Yolo-v8-Apex-Aim-assist](https://github.com/Franklin-Zhang0/Yolo-v8-Apex-Aim-assist)。向[Franklin-Zhang0](https://github.com/Franklin-Zhang0)和[Chalkeys](https://github.com/Chalkeys)致敬:thumbsup:。

## 性能水平
<details>
<summary>测试配置：</summary> 
    
    - OS: Windows 11 Pro 22H2
    - CPU: AMD Ryzen 3900X
    - GPU: NVIDIA GeForce RTX 2080 8G
    - RAM: 32G @ 3600MHz
    - Monitor resolution: 2560 x 1440
    - In-game resolution: 2560 x 1440
    - In-game FPS: Locked @ 60fps
    
</details>

| Model        | Size<br><sup>(pixels) | Enemy prediction<sup>val<br>mAP 50-95 | Teammate prediction<sup>val<br>mAP 50-95 | Speed<br><sup>.trt fp16<br>(fps) | Speed<br><sup>.trt int8<br>(fps) | IFF performence |
| -------------| --------------------- | -------------------- | -------------------- | ------------------------------ | ------------------------------- | ------------------ |
| Apex_8n      | 640                   | 54.5                 | 71.3                 | 58.88                          | 61.23                           | Poor                |
| Apex_8s      | 640                   | 66.9                 | 79.2                 | 27.10                          | 33.43                           | Acceptable               |
| Apex_8n_pruned| 640                  | 44.1                 | 55.9                 | 72.22                          | 74.04                           | Poor               |
| Apex_8s_pruned| 640                  | 52.9                 | 64.9                 | 30.54                          | 31.82                           | Poor               |


## Advantages and Todos
Advantages:
* [x] Fast screencapturing with dxshot
* [x] Not Logitech Macro/GHub dependent
* [x] Customizable trigger keys
* [x] PID capable
* [x] IFF(Identification friend or foe) capable (Works 50/50)
* [x] fp16 precision

Todos:
* [x] int8 precision (No significant improvement as for now)
* [x] Pruning (Sacrificies accuracy way too much)
* [ ] Increase accuracy of IFF
* [ ] Increase accuracy under: ADS, partial body exposure, gunfire blockage, smoke...

## 1. Set up the environment

### 1.1. Environment set up under Linux
- Install `Conda` (if not already installed)

    In your terminal window, run:`bash Miniconda3-latest-Linux-x86_64.sh` to install `Miniconda` (suggested) or run:`bash Anaconda-latest-Linux-x86_64.sh` to install `Anaconda`.
    
- Setup a new environment
    
    In your terminal window, run:
    ```shell
    conda create -n yolov8 python=3.10 # create environment 'yolov8' with python 3.10 installed 
    conda activate yolov8 # activate environment 'yolov8'
    ```
    
- Install `CUDA` and `PyTorch`.
   
   ~~if you have a cuda capable gpu, you can running the following extra command~~ 

  Running inferrence on CPU is suboptimal and no longer suggested by this project. Please use `CUDA` capable GPU and run the following command to install `CUDA`, `cuDNN` and `PyTorch`:
   ``` shell
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`
   ```
- Install `TensorRT`.
 
  Add the following command
   ```
   pip install --upgrade setuptools pip --user
   pip install nvidia-pyindex
   pip install --upgrade nvidia-tensorrt
   pip install pycuda
   ```
 - Install python requirement.
   ``` shell
   pip install -r requirement.txt
   ```
    

### 1.2. Environment set up in Windows10 & Windows11

The following method has being tested and successed under `Windows 10 Pro Version 21H2/22H2`, `Windows11 Pro Version 22H2` and `Windows11 Pro Insider View Build 25346`. Technically, it works under all latest `Windows OS` builds.
- Version-align

    |  CUDA   |  cuDNN   | TensorRT | PyTorch  |
    | :-----: | :------: | :------: | :------: |
    | 11.7.0  | 8.5.0    |  8.5.2.2 | 2.0.0    |
    | 11.8.0  | 8.6.0    |  8.5.3.1 | 2.0.0    |
    | ...    | ...   |  ... | ...    |
    
    We will be using the first row as our package manifests.
    
- Install `CUDA`. (One can also follow the official instruction:[`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)).
    ```shell
    conda install cuda -c nvidia/label/cuda-11.7.0 # install CUDA 11.7.0
    ```

- Install `cuDNN`.
    - Register for the [`NVIDIA developer program`](https://developer.nvidia.com/login).
    - Go to the cuDNN download site:[`cuDNN download archive`](https://developer.nvidia.com/rdp/cudnn-archive).
    - Click `Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x`.
    - Download `Local Installer for Windows (Zip)`.
    - Unzip `cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip`.
    - Copy all three folders (`bin`,`include`,`lib`) and paste them to the `CUDA` installation directory `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`. (NOTE `bin`,`include`,`lib` folders are already exist in the CUDA folder.).

- Install `PyTorch`.
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
- Install `TensorRT`.
    Follow the [Nvidia instruction of installation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip).
    - Go to the [TensorRT download site](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
    - Download `TensorRT 8.5 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package`.
    - Unzip the folder `TensorRT-8.5.2.2` from `TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6.zip`.
    - Add the `<your install path>\TensorRT-8.5.2.2\lib` into PATH environment variable.
    - Go to the folder `<your install path>\TensorRT-8.5.2.2\python`
    - In command window, input 
        ```shell
        conda activate yolov8 # activate dedicated environment
        pip install tensorrt-8.5.2.2-cp310-none-win_amd64.whl # install tensorrt package to python
        ```
 - Install python requirement.
   ``` shell
   pip install -r requirement.txt
   ```

<details>
<summary> Verify installation and check versions.</summary>
    
- Verify installation of `CUDA`, `cuDNN`, `PyTorch` and `TensorRT`.  
    
    - Verify `CUDA`.
        ```shell
        nvcc -V
        ```
        If installment successed, you should see prompts like:
        ```shell
        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2022 NVIDIA Corporation
        Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
        Cuda compilation tools, release 11.7, V11.7.64
        Build cuda_11.7.r11.7/compiler.31294372_0
        ```
     - Verify `cuDNN`.
     
        ```shell
        python
        import torch
        print(torch.backends.cudnn.version())
        ```
        
     - Verify `PyTorch`.
      
        ```shell
        python
        import torch
        print(torch.__version__)
        ```
    
    - Verify `TensorRT`.
      
        ```shell
        pip show tensorrt
        ```
        If installment successed, you should see prompts like:
        ```shell
        Name: tensorrt
        Version: 8.5.2.2
        Summary: A high performance deep learning inference library
        Home-page: https://developer.nvidia.com/tensorrt
        Author: NVIDIA Corporation
        ```    
    
    
    
</details>



## 2. Build your weight
### 2.1. PyTorch `.pt` weight
You have several options here to realize your `.pt` weight:
- (1) Use the provided weight `apex_v8n.pt`.
     
     This is a weight based on `yolov8n.pt`, trained with 7W screenshots and labeled 'Enemy', 'Teammate'. Due to the nature of this project, the provided weight is poorly trained to prevent abuse and cheating. However, it can already track characters well and demonstrate some level of IFF capability.  
     
- (2) Use the provided weight `apex_v8n.pt` as a pretrained weight and train your own weight with your own dataset.
     
     Please follow the [official instruction](https://docs.ultralytics.com/usage/cli/) of `Ultralytics` to train your own weight.
     
     Note that the dataset is required to use `YOLO` annotation format, please reform your dataset into the following structure:
        
    ```shell
    dataset/apex/
    ├── train
    |   ├── images
    │   |   ├── 000000000001.jpg
    │   |   ├── 000000580008.jpg
    │   |   |   ...
    |   ├── lables    
    │   |   ├── 000000000001.txt
    │   |   ├── 000000580008.txt
    │   |   |   ...
    ├── valid
    |   ├── images
    │   |   ├── 000000000113.jpg
    │   |   ├── 000000000567.jpg
    |   ├── lables    
    │   |   ├── 000000000113.txt
    │   |   ├── 000000000567.txt
    │   |   |   ...    
    ```
    
- (3) Train your own weight with official pretrained `yolov8` weights.
    
    If the provided weight which is based on `yolov8n.pt` can not meet your expectation. You can also explore the options of other pretrained weights provided by `yolov8`.
    
    - Model speed: `8n>8s>8m`
    
    - Model accuracy: `8n<8s<8m`
    
    Please follow the [official instruction](https://docs.ultralytics.com/usage/cli/) of `Ultralytics` to train your own weight.
    
### 2.2. ONNX `.onnx` weight (skip if only fp16 precision is desired)

You have several options here to to convert your `.pt` weight to a `.onnx` weight.

- (1) Use yolov8 built in function `YOLO export`:

    ```shell
    yolo export model=<your weight path>/best.pt format=onnx
    ```
    Note this built-in method is identical to the python code provided in [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
    
- (2) Use Paddleslim ACT (In Linux):
    
    

### 2.3. TensorRT `.trt` or `.engine` weight

Use any of the following methods to generate TensorRT engine:

- (1) Use yolov8 built-in function `YOLO export` to export `.engine` weight directly from `.pt` weight.
    
    ```shell
    # out put fp32 precision (default)
    yolo export model=<your weight path>/best.pt format=engine
    
    # out put fp16 precision (recommanded)
    yolo export model=<your weight path>/best.pt format=engine fp16=True
    ```
- (2) Use the third-party method [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) to export `.trt` weight from previous generated `.onnx` weight.

    ```shell
    # out put fp32 precision
    python export.py -o <your weight path>/best.onnx -e apex_fp32.trt -p fp32 --end2end --v8
    
    # out put fp16 precision (default, recommanded)
    python export.py -o <your weight path>/best.onnx -e apex_fp16.trt --end2end --v8
    
    # out put int8 precision (for extreme performance)
    python export.py -o <your weight path>/best.onnx -e apex_int8.trt -p int8 --calib_input <your data path>/train/images --calib_num_images 500 --end2end --v8
    ```
    
## 3. Run the program
Replace the model and change settings in the `args_.py` file (See Section 4.)

Run the `main.py` file with the following command
```shell        
conda activate v8trt # activate dedeicated env
python main.py # start aim assist
```
After a few seconds, the program will start fucntioning. You should see the follwing prompts in the console:
```shell
listener start
Main start
Screenshot fps:  311.25682541048934
fps:  61.09998811302416
interval:  0.016366615295410156
```

Explaination of keys:
- `Shift`: Holding shift triggers aim assist. By default, only `holding shift` can trigger the aim assist.
- `Left`: Unlock `left mouse button`. Once `left` key is clicked, you should hear a beep and now holding `left mouse button` can also trigger aim assist 
- `Right`: Unlock `right mouse button`. Once `right` key is clicked, you should hear a beep and now holding `right mouse button` can also trigger aim assist 
- `End`: Cilck `End` for continious aiming. Auto aiming is always on and another click to turn off.
- `Left mouse button`: Optional trigger
- `Right mouse button`: Optional trigger
- `Home`: Stop listening and exit program

## 4. Change settings
You can change following settings in the `args_.py` file.
- `--model`: The weight to be used by this project. Please replace this with your own `.trt` or `.engine` weight.
- `--classes`: classes to be detected, can be expanded but need to be an array. For example, 0 represents 'Teammate', 1 represents 'Enemy'. Then the input should be [1].
- `--conf`: Confidence level for inference. Adjust it based on your model accuracy.
- `--crop_size`: The portion to detect from the screen. Adjust it based on your monitor resolution. For example: `1/3` for 1440P, or `1/2` for 1080P.
- `--pid`: Use PID controller to smooth aiming and prevent overdrifting. Leave it by default is recommanded.
- `--Kp`,`--Ki`,`--Kd`: PID components that need to be carefully calibrated. `--Kp=0.5`, `--Ki=0.1`, `--Kd=0.1` is recommanded as starting point.
You can also modify the `MyListener.py` file.
- Function `def listen_key(key)` and def `keys_released(key)`: Change `keyboard.Key.home`, `keyboard.Key.end`, `keyboard.Key.shift`, `keyboard.Key.left` or `keyboard.Key.right` to whatever keys you like to customize the key settings.

## Reference
[Train image dataset](https://universe.roboflow.com/apex-esoic/apexyolov6)

[TensorRt code](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
