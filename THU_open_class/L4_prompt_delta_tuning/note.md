# L4_prompt_delta_tunnig

## 前提

本文环境为Windows11+only CPU；

算力不足，以下使用的模型为最少参数量的OPT-125M。

## 前置

[下载数据集](https://aristo-data-public.s3-us-west-2.amazonaws.com/arc-da/ARC-DA-v1.1.zip)

[下载模型](https://huggingface.co/facebook/opt-125m)

下载完后一起放到和optqa_to_complete.py一个目录下；

配好环境，下载opendelta和openprompt；

opendelta我建议你使用这个命令安装最新版，pip install git+https://github.com/thunlp/OpenDelta.git，如果使用pip install opendelta会出现由于sklearn已经弃用的错误。

## 流程

以下内容可以清楚明晰的看见openPrompt和openDelta分别应用于什么部分：

1. **加载数据集** pytorch的datasets

2. **加载预训练语言模型**（PLM）openPromt的load_plm可以用来加载预训练大模型，返回值：plm模型，tokenizer词元化处理器，model_config模型超参数，wrapper_class词元化后要包裹的类；

3. **添加deltaModel到大模型** openDelta的AutoDeltaConfig和AutoDeltaModel加载delta模型，然后把他嵌入进刚刚得到的plm中。

4. **初始化template** 使用openPrompt的ManualTemplate类进行实例化。

5. **训练模型** 使用promptModel进行训练，Q：那deltaModel呢？A：delta模块已经插入模型了，而prompt是在下游任务中附加句子中👼。

## 代码

代码由于篇幅关系，挪至相同目录下的optqa_to_complete.py中；

我的目录结构框架如下：

```
- 根目录
  - ARC-DA-v1.1
  	- dev,test,train.jsonl
  - delta_ckpts
  	- config.json
  	- pytorch_model.bin
  - opt-125m
  	- 模型文件
  - raw_delta_ckpts
  	- pytorch_model.bin
  - optqa_to_complete.py
```

## 测试

在我这个环境下delta_checkpoint的size约为3MB，而作为backbone的OPT模型是他的80倍；

使用参数进入预测模式 --mode interactive_inference，可以发现125M模型即使加上prompt learning以及delta tuning，精度也非常差👼。。

```
[INFO|(OpenDelta)saving_loading_utils:345]2024-02-08 23:22:56,706 >> Hash-check passed. You can safely use this checkpoint directly.
Input a question:Am I a father?
tokenizing: 1it [00:00, 676.50it/s]
begin evaluation
Answer: yes
Input a question:What is the cause of most earthquakes?
tokenizing: 1it [00:00, 1001.51it/s]
begin evaluation
Answer: the earth's crust is more stable
```