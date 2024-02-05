# L2_NLP_pipeline_pytorch

## 提示

后面的指导不再涉及关于conda虚拟环境创建；

先理解课程附带代码。

## 项目初始化

首先，我们可以尝试运行一下运行课程附带代码：

​	python main.py -h (查询相关参数)

​	python main.py

## 代码

需要完成的目标：

- 完成二分类情感判断任务，根据句子判断是positive还是negative；
- 完成main.py data.py model.py三个文件的编写；
- 可以修改给出的LSTM代码。

情感分类任务参考代码在同目录（仅供参考。。。👼）

## 训练

参考结果如下：

​	注意，改了一个batch的数据存取，把bptt设置为1，lr设置为0.0001，其他超参数默认。

	| epoch   1 |   200/ 3367 batches | lr 0.0001 | ms/batch 54.74 | loss  0.69 | ppl     2.00
	| epoch   1 |   400/ 3367 batches | lr 0.0001 | ms/batch 56.54 | loss  0.68 | ppl     1.98
	| epoch   1 |   600/ 3367 batches | lr 0.0001 | ms/batch 54.49 | loss  0.69 | ppl     1.98
	| epoch   1 |   800/ 3367 batches | lr 0.0001 | ms/batch 57.40 | loss  0.62 | ppl     1.86
	| epoch   1 |  1000/ 3367 batches | lr 0.0001 | ms/batch 55.40 | loss  0.56 | ppl     1.74
	| epoch   1 |  1200/ 3367 batches | lr 0.0001 | ms/batch 54.71 | loss  0.51 | ppl     1.67
	| epoch   1 |  1400/ 3367 batches | lr 0.0001 | ms/batch 54.46 | loss  0.48 | ppl     1.62
	| epoch   1 |  1600/ 3367 batches | lr 0.0001 | ms/batch 53.92 | loss  0.46 | ppl     1.58
	| epoch   1 |  1800/ 3367 batches | lr 0.0001 | ms/batch 54.31 | loss  0.46 | ppl     1.59
	| epoch   1 |  2000/ 3367 batches | lr 0.0001 | ms/batch 54.40 | loss  0.43 | ppl     1.54
	| epoch   1 |  2200/ 3367 batches | lr 0.0001 | ms/batch 54.85 | loss  0.42 | ppl     1.52
	| epoch   1 |  2400/ 3367 batches | lr 0.0001 | ms/batch 53.84 | loss  0.41 | ppl     1.51
	| epoch   1 |  2600/ 3367 batches | lr 0.0001 | ms/batch 55.28 | loss  0.41 | ppl     1.50
	| epoch   1 |  2800/ 3367 batches | lr 0.0001 | ms/batch 54.78 | loss  0.40 | ppl     1.49

​		训练准确率为 74.77064220183486 % accuracy

