# Yolov8 Aim Assist with IFF

## 性能水平

  <details>
  <summary>测试配置：</summary> 

    - 操作系统: Windows 11 22H2
    - 处理器: AMD Ryzen 3900X
    - 显卡: NVIDIA GeForce RTX 2080 12G
    - 内存: 16G @ 3600MHz
    - 屏幕分辨率: 2560 x 1440
    - 游戏分辨率: 2560 x 1440
    - 测试游戏: Apex
    - 游戏内FPS: 60+fps

  </details>

| 模型             | 图像大小px | 敌方识别度mAP 50-95 | 友方识别度mAP 50-95 | .trt fp16速度(fps) | .trt int8速度(fps) | IFF性能 |
|----------------|--------|----------------|----------------|------------------|------------------|-------|
| Apex_8n        | 640    | 54.5           | 71.3           | 58.88            | 61.23            | 差     |
| Apex_8s        | 640    | 66.9           | 79.2           | 27.10            | 33.43            | 良好    |
| Apex_8n_pruned | 640    | 44.1           | 55.9           | 72.22            | 74.04            | 差     |
| Apex_8s_pruned | 640    | 52.9           | 64.9           | 30.54            | 31.82            | 差     |

## 优点和待办

  优点：

  * [x] 使用dxshot快速截屏
  * [x] 不依赖于Logitech Macro/GHub
  * [x] 可自定义的触发键
  * [x] PID控制算法
  * [x] 敌我识别模型
  * [x] fp16精度

  待办：

  * [x] int8精度（目前没有显著改进）
  * [x] 裁剪（牺牲精度太多）
  * [ ] 敌我识别提高准确性
  * [ ] 部分身体暴露、炮火堵塞、烟雾等场景提高精度

## 环境搭建

### 1. Linux系统

  - 安装`Conda`
  
    使用`bash Miniconda3-latest-Linux-x86_64.sh`来安装`Miniconda`，或者使用`bash Anaconda-latest-Linux-x86_64.sh`
    来安装`Anaconda`
  
  - 创建虚拟环境
  
    ```shell
    conda create -n yolov8 python=3.10 # 指定py版本创建conda虚拟环境
    conda activate yolov8 # 激活环境
    ```
  
  - 安装`CUDA`和`PyTorch`
  
    如果你的电脑有独显，可以安装CUDA、cuDNN、PyTorch，以达到更流畅的效果
    ``` shell
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`
    ```

  - 安装 `TensorRT`

    ``` shell
    pip install --upgrade setuptools pip --user
    pip install nvidia-pyindex
    pip install --upgrade nvidia-tensorrt
    pip install pycuda
    ```

  - 安装其他依赖库

    ``` shell
    pip install -r requirement.txt
    ```

### 2. Windows系统

  `Windows 10 Pro Version 21H2/22H2`、`Windows11 Pro Version 22H2`、`Windows11 Pro Insider View Build 25346`、`Windows OS`
  已在以上版本进行了测试并取得了成功，从技术上讲，它适用于所有最新版本

  - 版本选择

  |  CUDA  | cuDNN | TensorRT | PyTorch |
  |:------:|:-----:|:--------:|:-------:|
  | 11.7.0 | 8.5.0 | 8.5.2.2  |  2.0.0  |
  | 11.8.0 | 8.6.0 | 8.5.3.1  |  2.0.0  |
  |  ...   |  ...  |   ...    |   ...   |
  后面将使用第一行版本清单

  - 安装`CUDA`

    访问[`CUDA官网文档`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)，命令安装
    ```shell
    conda install cuda -c nvidia/label/cuda-11.7.0 # 安装CUDA 11.7.0
    ```
    或者访问[`CUDA官网下载`](https://developer.nvidia.com/cuda-downloads)，手动下载解压并配置环境变量，具体操作可看[【深度学习之YOLO8】环境部署](https://jinke.love/blog-deeplearning/15.html)，有CUDA、cuDNN和PyTorch的详细安装教程

  - 安装`cuDNN`

    - 注册并成为[NVIDIA开发人员](https://developer.nvidia.com/login)
    - 转到cuDNN下载站点：[cuDNN下载存档](https://developer.nvidia.com/rdp/cudnn-archive)
    - 点击`Download cuDNN v8.5.0 (August 8th, 2022), for CUDA 11.x`
    - 下载`Local Installer for Windows (Zip)`
    - 解压缩`cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip`
    - 复制所有三个文件夹(`bin`/`include`/`lib`)并将它们粘贴到CUDA安装目录(CUDA默认安装目录`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`)

  - 安装`PyTorch`

    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

  - 安装`TensorRT`

    [按照英伟达的安装说明进行操作](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)
    - 转到[TensorRT下载站点](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
    - 下载`TensorRT 8.5 GA for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8`压缩包
    - 压缩下载的文件`TensorRT-8.5.2.2TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6.zip`
    - 将`xxxx\TensorRT-8.5.2.2\lib`目录添加到环境变量
    - cmd 进文件夹`xxxx\TensorRT-8.5.2.2\python`，运行如下命令
  
    ```shell
    conda activate yolov8 # 激活环境
    pip install tensorrt-8.5.2.2-cp310-none-win_amd64.whl # 使用.whl文件快速安装tensorrt
    ```
  
  - 安装其他依赖库

    ``` shell
    pip install -r requirement.txt # requirement.txt在项目的根目录
    ```

  <details>
  <summary>验证安装版本</summary>

  - 验证`CUDA`
  
    ```shell
    nvcc -V
    ```
    如果`CUDA`安装成功，它会显示类似如下结果：
  
    ```shell
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
    Cuda compilation tools, release 11.7, V11.7.64
    Build cuda_11.7.r11.7/compiler.31294372_0
    ```

  - 验证`cuDNN`

    ```shell
    python
    import torch
    print(torch.backends.cudnn.version())
    ```

  - 验证`PyTorch`

    ```shell
    python
    import torch
    print(torch.__version__)
    ```

  - 验证`TensorRT`
    ```shell
    pip show tensorrt
    ```
    如果`TensorRT`安装成功，它会显示类似如下结果：
  
    ```shell
    Name: tensorrt
    Version: 8.5.2.2
    Summary: A high performance deep learning inference library
    Home-page: https://developer.nvidia.com/tensorrt
    Author: NVIDIA Corporation
    ```

</details>

## 准备权重

### 1. PyTorch`.pt`权重

  您有几种选择来制作权重文件：

  - 使用已经做好的权重`apex_v8n.pt`

  这是一个基于权重，使用7W截图训练并标记为"敌人"、"队友"，由于该项目的性质，提供的体重训练不足，以防止滥用和作弊，但是，它已经可以很好地跟踪角色并展示一定程度的IFF功能

  - 使用提供的权重作为预训练的权重，并按照自己的数据集训练自己的权重

  请按照[Ultralytics官方说明](https://docs.ultralytics.com/usage/cli/)训练自己的体重

  请注意，数据集需要使用标注格式，请将您的数据集修改为以下结构：
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

  - 用官方预训练的重量训练自己的重量

  如果提供的重量不能满足您的期望，您还可以探索提供的其他预训练权重的选项
  <br>模型速度：8n>8s>8m
  <br>模型精度：8n<8s<8m
  <br>请按照[Ultralytics官方说明](https://docs.ultralytics.com/usage/cli/)训练自己的权重模型

### 2. `.pt`导出为`ONNX`的`.onnx`权重

  如果你只需要`fp16`精度，那么跳过此步

  [官方导出文档](https://docs.ultralytics.com/usage/python/#export)
  [官方导出参数](https://docs.ultralytics.com/usage/cfg/#export)

  - CLI官方函数导出`yolo export`

    ```shell
    yolo export model=C://user/zjk/xxx.pt format=onnx
    ```
  - Python代码导出

    ```shell
    from ultralytics import YOLO
    # 加载模型
    model = YOLO("./models/yolov8n.pt")  # 加载预训练模型（建议用于训练）
    success = model.export(format="onnx",device=0)  # 将模型导出为 ONNX 格式
    # success = model.export(format="engine",device=0)  # 将模型导出为 TensorRT 格式
    print("ok")
    ```

### 3. `.pt`导出为`TensorRT`的`.trt`、`.engine`权重

  方法同转ONNX权重

  - CLI官方函数导出`yolo export`

    ```shell
    # 默认输出为fp32精度
    yolo export model=C://user/zjk/xxx.pt format=engine
    # 指定输出为fp16精度
    yolo export model=C://user/zjk/xxx.pt format=engine fp16=True
    ```
  
  - 特殊地，`.onnx`导出为`.trt`权重

    ```shell
    # 转fp32精度
    python export.py -o <your weight path>/best.onnx -e apex_fp32.trt -p fp32 --end2end --v8
    # 转fp16精度
    python export.py -o <your weight path>/best.onnx -e apex_fp16.trt --end2end --v8
    # 转int8精度
    python export.py -o <your weight path>/best.onnx -e apex_int8.trt -p int8 --calib_input <your data path>/train/images --calib_num_images 500 --end2end --v8
    ```

## 运行

  ```shell        
  conda activate `你的conda环境名`
  python main.py
  ```
  或者使用PyCharm等工具，配置环境后运行main文件

  几秒钟后，程序将开始功能化。您应该会在控制台中看到以下提示
  ```shell
  listener start
  Main start
  Screenshot fps:  311.25682541048934
  fps:  61.09998811302416
  interval:  0.016366615295410156
  ```
<details>
<summary>按键说明</summary>

  - Shift：按住换档触发瞄准辅助。默认情况下，只能触发瞄准辅助。holding shift
  - 向左：开锁。单击键后，您应该会听到哔哔声，现在 保持也可以触发瞄准辅助
  - 向右：开锁。单击键后，您应该会听到哔哔声，现在 保持也可以触发瞄准辅助
  - End：用于连续瞄准的西克。自动瞄准始终处于打开状态，再次单击可关闭。End
  - 左键：可选触发器
  - 右键：可选触发器
  - Home：停止收听并退出程序

</details>



## 配置自定义

  修改`args_.py`文件以得到自己想要的效果
  
  - model：此项目权重文件是必要的。您可以用自己的`.trt`、`.engine`权重文件代替它
  - classes：要检测的类，可以扩展，但需要是一个数组。例如，0表示"队友"，1代表"敌人",那么输入应该是 [1]
  - conf：推理的置信度,根据您的模型精度进行调整
  - crop_size：要从屏幕检测的部分,根据您的显示器分辨率进行调整,例如：对于1440P或1080P,1/31/2
  - pid：使用PID控制器平滑瞄准，防止过漂移,默认情况下保留它被重新建议，需要仔细校准的PID组件，您也可以修改该文件和`MyListener.py`
  - 功能和定义：无论您喜欢什么键，都可以自定义键设置`listen_key(key)`、`keys_released(key)`等

## 免责声明

  - 该项目无意用于游戏，也不应该使用。使用风险自负
  - 该项目将不会定期维护，因为它目前已经达到了目标。一旦有更好的模型/压缩方法/量化方法可用，可能会提供更新
  - 该项目是基于[Chalkeys/Yolov8-Apex-Aim-assist-with-IFF](https://github.com/Chalkeys/Yolov8-Apex-Aim-assist-with-IFF)
    项目的翻译和改进，原项目[Franklin-Zhang0/Yolo-v8-Apex-Aim-assist](https://github.com/Franklin-Zhang0/Yolo-v8-Apex-Aim-assist)
    。向[Franklin-Zhang0](https://github.com/Franklin-Zhang0)和[Chalkeys](https://github.com/Chalkeys)致敬:thumbsup:

## 资料

  [Train image dataset](https://universe.roboflow.com/apex-esoic/apexyolov6)
  
  [TensorRt code](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
