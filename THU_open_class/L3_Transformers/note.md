# L3_Transformers

## 提示

Fine tuning指的是利用数据集再对你需要的下游任务进行一次训练；

每个checkpoint保存失败，继续看下文。

## 代码

根据demo提供的模板，我们同样将代码分为：

- 数据集下载
- tokenization
- 模型下载
- 设定参数
- 训练

### 数据集下载

```python
from datasets import load_dataset, load_metric
dataset = load_dataset("glue", "qnli")
metric = load_metric("glue", "qnli")
```

### Tokenization

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=512)
encoded_data = dataset.map(preprocess_function,batched=True)
```

### 模型下载

```python
model = AutoModelForSequenceClassification.from_pretrained("google/bert_uncased_L-2_H-128_A-2",num_labels=2)
```

### 设定参数

```python
from transformers import TrainingArguments

batch_size = 16

args = TrainingArguments(
    output_dir="./result",
    evaluation_strategy="epoch", # 在每个epoch结束后测试效果
    save_strategy="epoch", # 每个epoch结束后保存模型
    learning_rate=2e-5, # 学习率
    per_device_train_batch_size=batch_size, # 每个GPU训练的batch size
    per_device_eval_batch_size=batch_size, # 每个GPU测试的batch size
    num_train_epochs=5, # 训练的epoch数
    weight_decay=0.01, # 权重衰减
    load_best_model_at_end=True, # 训练结束后，是否加载在验证集上表现最好的模型
    metric_for_best_model="accuracy" # 准确率为指标
)
```

### 训练

```python
from transformers import Trainer
import numpy as np
def compute_metrics(eval_pred):
    logits, lables = eval_pred
    predictions = np.argmax(logits,axis=1)
    return metric.compute(predictions=predictions, references=lables)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_data["train"],
    eval_dataset=encoded_data["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
trainer.train()
```

### 保存失败

这个问题也是困扰了我一两个小时，然后我在huggingface forum上找到一个[[Permission issues when saving model checkpoint - Beginners - Hugging Face Forums](https://discuss.huggingface.co/t/permission-issues-when-saving-model-checkpoint/70109/2)]，但是不太清楚原理，但是有用。

## 结果

这里我的环境只有CPU，跑的时间非常长👼；

```
{'eval_loss': 0.650716245174408, 'eval_accuracy': 0.6172432729269632, 'eval_runtime': 8.4368, 'eval_samples_per_second': 647.518, 'eval_steps_per_second': 40.537, 'epoch': 5.0}
```

```
TrainOutput(global_step=32735, training_loss=0.6398471140653937, metrics={'train_runtime': 2893.0922, 'train_samples_per_second': 181.023, 'train_steps_per_second': 11.315, 'train_loss': 0.6398471140653937, 'epoch': 5.0})
```

通过trainer.evaluate()输出的结果；

```
{'eval_loss': 0.650716245174408,
 'eval_accuracy': 0.6172432729269632,
 'eval_runtime': 9.1348,
 'eval_samples_per_second': 598.046,
 'eval_steps_per_second': 37.439}
```

附加，通过pipeline，我们可以尝试预测test集上的label；

```python
from transformers import pipeline

# Load the pre-trained model
classifier = pipeline("text-classification",model="result/checkpoint-32735")

# predictions = classifier(encoded_data["test"][0]["question"]+encoded_data["test"][0]["sentence"])

predictions = []

for item in encoded_data["test"]:
    predictions.append(classifier(item["question"]+item["sentence"]))
    
output_file = "QNLI.tsv"

with open(output_file, "w") as f:
    f.write("index\tprediction\n")
    for i,item in enumerate(predictions):
        pred_str = "not_entailment" if (item[0]["label"] == "LABEL_0") else "entailment"
        f.write(f"{i}\t{pred_str}\n")

```

代码下面我还从官方下载了QNLI的示例提交数据，但是官方并没有说明这一份提交的标签是否完全与答案一致，所以仅供参考。
