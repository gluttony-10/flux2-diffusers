# flux2-diffusers
flux2的diffusers版本

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 使用需求
1.显卡支持BF16

2.显存大于4G

## 安装依赖
```
git clone https://github.com/gluttony-10/flux2-diffusers
cd Tongbi
conda create -n Tongbi python=3.12
conda activate Tongbi
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
## 下载模型
```
modelscope download --model Gluttony10/flux2-diffusers --local_dir ./models
```
## 开始运行
```
python glut.py
```
## 参考项目
https://github.com/black-forest-labs/flux2
