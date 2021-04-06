# bert_of_thesues
采用bert_of_thesues论文里的蒸馏方法，在ner任务上做的蒸馏实验

```
python == 3.7
pytorch == 1.6.0
transformers == 3.4.0
```

### Data
* 首先将数据处理成`BIO`格式，processed文件夹下存放的是命名实体识别的数据，代码可参考`data_process.ipynb`
* 下载中文BERT预训练模型,来自`transformers`

### 模型训练

```
sh ./scripts/run_predecessor.sh  先在具体任务上fine-tune
sh ./scripts/run_prune.sh  再进行蒸馏
sh ./scripts/run_successor.sh  最后在student模型上继续fine-tune
```


### 模型预测
```
python crf_predict.py
```


