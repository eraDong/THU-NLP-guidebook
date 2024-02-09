# L5_BMSystem

## 前提

BMTrain 大模型并行计算框架

- 数据并行

  仅仅对数据切分，其他因素保持。

  *操作算子：*

  1. Broadcast 将一张显卡上的数据传输到其他显卡；
  2. Reduce 将所有显卡的数据进行（求和/平均），传到一张显卡；
  3. All Reduce 将所有显卡的数据进行（求和/平均），传到所有显卡；
  4. Reduce Scatter 将所有显卡的数据进行（求和/平均），分批传到所有显卡；
  5. All Gather 与Reduce Scatter配合，收集所有显卡的数据，然后发送到所有显卡。

  *分布式数据并行：*

  ​	舍弃参数服务器，自己进行更新。

- 模型并行

  对参数，梯度，优化器切分，保持数据不变；

  矩阵的子结果拼接成最终结果。

- ZeRO

  以上两种方法都有各自的pros and cons；

  1. ZeRO-1 stage：基于数据并行，但是每张显卡只获得部分梯度Reduce Scatter去更新部分参数，然后通过All Gather把参数给到每张显卡上；
  2. ZeRO-2 stage：得到分批的Gradient*之后，则删除没有分批处理的梯度；
  3. ZeRO-3 stage：只保存部分参数，需要用到所有参数的时候使用All Gather。

- 流水线并行

  把模型不同的层分给不同的显卡。

- Tricks

  1. 混合精度；
  2. offloading；
  3. overlapping；
  4. checkpointing。

BMCook 大语言模型压缩框架

- 知识蒸馏；
- 模型减枝；
- 模型量化；
- 其他：参数共享，低秩分解，新的模型结构。

BMInf 平民设备运行大模型框架

量化：缩放+INT；

虚拟内存。

## 作业

根据readme中的github指导即可，基本上一步到位。