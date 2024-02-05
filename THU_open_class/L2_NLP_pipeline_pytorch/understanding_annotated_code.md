# 理解课程附带代码

## 初始化

### main.py

argparse 用于在控制台中添加参数，可以更好地利用控制台添加/修改参数去训练模型；

torch.manual_seed 设置特定的种子，固定随机性，使得实验结果具有可重复性；

cuda 如果有GPU就会开启GPU训练。

## 数据预处理

### data.py

Dictionary类

- 初始化，创建了word2idx dict，idx2word list；
- 增加单词，这里使用了比较简单的线性映射，如果单词不存在于dict中，那么就会把单词放进list，dict就把这个单词映射为list目前的长度-1。

Corpus类

- 定义一个Dictionary，分别词元化train.txt，valid.txt，test.txt；
- 打开文本文件，对于每一行句子用空格分割单词并在结尾加上 &lt;eos&gt;标志，然后把所有单词都加入到Dictionary里面。
- 然后处理为tensor形式，创建一个ids list以及一个idss list，同样分割句子为每个单词，添加每个单词在Dictionary映射后的数字到ids中，处理完一行后把ids tensor化加入到idss中；
- 最后把所有concat起来，return结果。

### main.py

批量化数据，去掉数据中不能完整构成一列特征的列；

data.narrow(dim,start,len) 按纬度从start截取len的数据；

data.view(column,row) 重新排列数据然后转置，每一列代表一个batch。

 ## 训练

从epoch循环顺序往下：

​	train()

​		把model设置为train模式，适应于dropout，然后初始化隐藏层；

​		迭代每一个batch的数据，清零梯度，repackage_hidden获取隐藏层的值，然后前向传播计算损失值，反向传播计算梯度，clip_grad_norm防止梯度爆炸，更新参数。

​		输出训练数据。

​	evaluate()

​		评测训练出来的模型，换成eval模式，初始化隐藏层；

​		取数据，前向传播计算损失加起来，返回值。

最后是一些优化/退出/中断处理。

