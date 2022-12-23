# TSAD





## File description

- main.py 主函数
- options.py 参数设置函数
- src/dataloader.py: 数据装载函数
- src/loss.py: 损失函数
- src/model_decoder.py: 存放各种负责decoder的模型，包括重构模块
- src/model_embed.py: 存放embedding layer，主要目的是将原始的时序维度 (Batchsize\*SeqLength\*Dim) 变成 (Batchsize\*SeqLength\*EmbedDim)
- src/model_tcn.py: TCN编码器
- src/model_transformer: Transformer编码器
- optimizer.py: 优化器
- runner.py: 训练和测试的代码
- tools.py: 其他小工具
- save/model: 存放训练好的模型


## 模型运行

例如运行SMD数据集，选择TCN编码器，训练Epoch设置为10，运行如下代码

`python -m main -d SMD `