基于汉字笔画部首的多模态OCR
目标：可以实现zeroshot识别不同字体的汉字。
思路大致如下：
1. 汉字拆分成笔画输入。
2. 汉字的图片变成VITtoken
3. 通过MLM的方式把汉字进行对齐。