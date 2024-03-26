Open cmd as admin
python=3.9, NOTE: v3.9 can't install pyaudio, which is required by EfficientWord-net
v3.10 can't install tflite-runtime also required by EfficientWord-net,
so we change it's requirement.txt to support higher pyaudio version

1. Install wenet(https://github.com/wenet-e2e/wenet)
   pip install git+https://github.com/wenet-e2e/wenet.git
2. install EfficientWord-Net(https://github.com/Ant-Brain/EfficientWord-Net)
   before performing
   pip install EfficientWord-Net
   must perform
   mamba install anaconda::pyaudio
   first. EfficientWord-net needs pyaudio 0.2.11, but pip can't install its dependency portaudio for 0.2.11. (The latest version has no problem). NOTE:python v3.9 can't install pyaudio,
   Using mamba can install dependency correctly
3. Install EfficientWord-Net from local source (Recommand, the original code has compatiable issues)
   prerquist: install pyaudio mentioned in item 2
   install tflite_runtime: python -m pip install tflite-runtime 或是 pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
   a. clone repo from https://github.com/shaominFei/EfficientWord-Net.git, the path should not have Chinese character
   b. activate virtual env, then go to the directory where setup.py sits, run
   python -m pip install .

总结：

1. python=3.9
2. install wenet
3. install pyaudio,
4. 安装 tflite-runtime
5. 安装 EfficientWord-Net
   目前版本是 0.2.14, 需要从本地安装 EfficientWord-Net, 需要修改它的 requirements.txt 中的版本依赖
   numpy>=1.22.0
   PyAudio>=0.2.11
   requests==2.26.0
   onnxruntime>=1.14.1


   install on untuntu:
   python==3.9
   1. go to src folder
   2. run pip install -r requirements.txt
   3. clone wenet
   git clone https://github.com/wenet-e2e/wenet.git
   cd wenet
   git tag -l
   git checkout tags/<tag_name>
   now the tag_name is v3.0.1
   4. install wenet from local source code
   go to wenet folder and run:
   python -m pip install .
   5. install EfficientWord-new from local source, as mentioned above
   a. conda install anaconda::pyaudio
   b. python -m pip install tflite-runtime
   c. go to the local source code folder, run
   python -m pip install .
   6. install paddle 2.6
   https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html
   
   python -m pip install paddlepaddle==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
   7. install WeTExtProcessing to take care of number-text convertion
    pip install WeTextProcessing
