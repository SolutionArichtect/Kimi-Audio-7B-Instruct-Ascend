# Kimi-Audio (Ascend NPU Version)

该项目基于 [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) 进行修改，适配华为昇腾 NPU 环境。

## 模型准备

安装 `modelscope` 并下载模型：

```bash
pip install modelscope
modelscope download --model moonshotai/Kimi-Audio-7B-Instruct --local_dir ./{你的模型目录}
modelscope download --model ZhipuAI/glm-4-voice-tokenizer --local_dir ./{你的模型目录}
```

请同步修改以下文件中的模型路径：
- `cog-Kimi-Audio-7B-Instruct-main/kimia_infer/api/prompt_manager.py`
- `cog-Kimi-Audio-7B-Instruct-main/infer.py`

## 快速开始

### 1. 使用容器镜像

#### 1.1 使用成品镜像
拉取镜像（详情请咨询作者）：
```bash
docker pull {镜像名称}
```

运行推理：
```bash
# 基础推理
python infer.py

# 支持多轮对话
python infer1.py
```

### 2. 自制环境

**基础镜像**：`quay.io/ascend/vllm-ascend:v0.11.0rc0`
[镜像链接](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=v0.11.0rc0)

启动容器命令：
```bash
docker run \
--privileged \
--name kimi \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /home/data:/workspace/model \
-p 8200:8200 \
-it quay.io/ascend/vllm-ascend:v0.11.0rc0 bash
```

> **注意**：`-v /home/data:/workspace/model` 建议挂载您的模型目录。

#### 环境配置

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   （也可参考原项目链接中的 `requirements.txt` 按需安装）

2. **安装 FFmpeg**
   ```bash
   wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
   tar -xvf ffmpeg-4.0.1.tar.bz2
   mv ffmpeg-4.0.1 ffmpeg
   cd ffmpeg
   ./configure --enable-shared
   make -j 64
   make install
   ```

3. **安装 Decord**
   ```bash
   git clone --recursive https://github.com/dmlc/decord
   cd decord
   if [ -d build ];then rm build;fi && mkdir build && cd build
   cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
   make -j 64
   make install
   cd ../python
   python3 setup.py install --user
   ```

4. **FlashAttention**
   FA 算子已替换为昇腾亲和算子，根据项目需求安装对应依赖即可。
