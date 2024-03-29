
# 
## Data 繁体字识别，ocr的一些数据集

### 完成
- [x] Data/1954-01-02 校验完毕
### todo

# 2023年06月01日
无论怎样，OCR的训练的数据集终于开始启动了。
自从开始此项目依赖，脑子中想到的几个想法：
0. 希望把国务院公告，由图片，转换成格式化文本，之后在上面大家各种应用。
## OCR阶段
1. 最开始的想法是通过zi2zi的方式，原始文档的字符训练好。开始接触到gan。
2. 遇到的主要为题就是字符的切割的问题，需要拿单独的字符进行训练。
3. zi2zi的效果不好，主要是泛化性的问题，最开始的方案是，对于同一个汉字，一个视觉坏的字符对应一个字体输出的标准汉字，需要找出所有的汉字对。
4. 为了找到比较好的汉字对，最开始使用的规则进行切割，主要是水平方向的投影和数直防线的投影，比较难以解决两个字符距离比较近的情况和单个字符中间有空隙，比如汉字：北。
5. 之后尝试使用神经网络的方法，把汉字切割成小的图片，宽度为16像素高度为64的图片，通过resnet的方式对图像进行扫描，预测0或者1，0表示16像素的宽度位于汉字之内，1表示位于汉字之间，最开始标注数据10句正确率就可以达到95%左右，虽着标注数量的增多，准确率大概开始下降，随着标注数据的增加，大概标注了65个句子，（约2w+的切片，）而且训练5到6之后，模型的预测效果非常好，基本上每个句子改几个错误就可以形成新的数据集，就可以达到f1为94.5%的效果。
6. 基于神经网络预测两个汉字之间的0，1分类（汉字的切割）能否提升？出现的大部分错误是，对于两个汉字之间和汉字内部16像素比较容易出现混淆，随着模型看到数据越多，模型越难以区分，对于人也难以区分，一张16*64的图片可能是两个汉字之间的，如果未来要改善的话，第一个把像素16宽度加宽32，48，第二个的方案是序列化，目的都是可以使模型看到可以看到更多的上下文信息。尝试使用过gru模型进行预测，大概是在真确率86%左右，简单分析的原因是数据量不够大。
7. 大概是3月中下旬，确认了zi2zi效果不好(当时拿了1980个字进行训练，然后在整个数据集上跑，出现30%左右的汉字根本看不清的情况)，分析一种原因是字符切割的不好，4月份写了神经网络验证能否辅助字符切割，另外是想通过后期文本改正的方式，读了爱奇艺和清华大学的两篇论文和代码。
## 格式化文本阶段
8. 4月主要的精力是放到layoutLM的相关论文和代码复现上，基本上掌握了transform模型和transforms库，尝试了但是使用layoutLMv3对国务院公告进行版本分析，效果不算错，但是预训练模型阶段没有加入汉字模态的信息，根据abode的论文，如果可以加入汉字模态的话，效果可以更好，我也希望可以解决预测过程中：1重复预测的问题，一个大的区域预测的是正文，里面嵌套一个小的区域也是正文。 2 小区域文本预测错误的问题。例如落款和抬头预测的问题，经常会弄混，希望加入语义信息之后，预测效果会更好一些。
9. 主要输出的东西，就是完成了layoutlmv3的代码重构和打通label-studio和layoutlmv3的联调工作。
## OCR工作
10. 五月初确认完成layoutlmv3的代码和工程实践后，确定了如果完成数据格式化工作还是需要，训练ocr，考虑到zi2zi的效果比较差，因此采用了cg-gan作为字体风格迁移的工作。
11. 基本掌握了cg-gan模型结构和代码，开始尝试使用，把不好的字体变好，5月29日尝试了第一次大规模的训练，生成了1w张图片进行微调。训练数据集上90%，效果不好不是特别理想。
12.  5月30日，改变了思路，把好的汉字变坏，即好的字体学习坏的字体的风格。成功了，5月31一日，准备300w行的训练数据（大概4000万字），问题在于第一数据量太大，20w行数据的图片，大概为29G，所以目前采用的方案是先训练10w个数据，在V10032G上进行训练800个epoch，需要12day才能训练完，只要效果不错：加大数据，预测的结果可以效果更好，那就证明我的方法是可以行的通的。

**2023-06-01 03:28:36 ，800个epoch训练完成了5个，虽然acc依然是0，但是可以观察到loss在不断的降低。所以面试中就可以说清楚，我做不到99%的准确率是因为训练成本和时间的原因，我的思路是可行的**

**未来的工作就是查漏不缺，完善一下LayoutLM的部分工作**


## 2023-06-03 
ocr训练结果出来，效果非常好，粗侧准确率96+以上。

以下是对比：
 pdocr 原來的模型              新模型：  2023-06-03
  尚未顏髮不材場管理辦法之地區 -> 尙未頒發木材巿場管理辦法之地區,
  市政府應肥根據本地區情況 -> 市政府應即根據本地區情况
  木材市場韻頜導機構。  -> 木材巿場的頜導機構
 '裕一九五五年的農業增產創造有利條件 -> 給一九五五年的農業增產创造有利條件 
       查地北鎮捆                       ->  各地必須抓 
----
在百度的V100上训练大概42个小时，成本在80-120左右，需要进一步提升的特例和标点符号，预计需要训练8个小时。可能在需要2-3轮，主要的花费时间是找到特例对应的图片。

# 从0到1的OCR技术方案

## 背景

## 遇到的问题

## 技术方案

### 整体方案介绍
### 字符串切割
#### 对比学习
#### MLP和GRU

### 风格迁移
#### 端到端的风格迁移
#### 字to字的方案

### 数据集生成

## 模型训练

### 训练方案

### 特例情况

## 总结

## 附录
基于字符切割的方案。
参考文献