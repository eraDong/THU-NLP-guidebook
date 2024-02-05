# L1_Word2Vec_hpsearch

## 提示

出于对时间成本的考虑，实验强烈建议使用具有GPU的环境进行训练；

若出现运行错误，可以先尝试使用ChatGPT排查。

## 项目初始化

首先我们需要将远程仓库clone到本地：

​	git clone https://github.com/OlgaChernytska/word2vec-pytorch

接着对项目进行初始化，

根据课程提示，我们对每一个项目都创建一个虚拟环境：

​	conda create -n l1word2vec python=3.6

​	cd word2vec-pytorch

​	pip install -r requirements.txt

## 训练

编写简单bash脚本运行代码：

​	由于本文于Windows进行作业练习便忽略此操作。

利用Vscode手动运行模型训练：

​	在文件页面修改配置config.yaml：

- model_name "skigram" "cbow" 设置使用的模型；
- dataset "WikiText2" "WikiText03" 设置用于训练的数据集；
- model_dir 设置实验结果目录应该以"weights/开头"。

​	在终端页面：

​		conda activate l1word2vec

​		python train.py --config config.yaml

训练结果：

​	Train Loss=4.65033, Val Loss=4.69101

## 修改超参数

根据课程提示，我们调整一下config.yaml的learning_rate和train_batch_size并记录最后一个批次的loss和相应的超参数值：

​	这里我选择的数值如下：

​		train_batch_size: 64

​		learning_rate: 0.03

训练结果：

​	Train Loss=4.62812, Val Loss=4.69273

​		

​	

