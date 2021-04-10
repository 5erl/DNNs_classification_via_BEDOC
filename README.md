Repository of supplementary materials for paper published in journal MDPI Applied Sciences.

### Deep Neural Networks Classification via Binary Error-Detecting Output Codes

This repository makes available all trained neural networks published in the paper and part of source codes by which
results presented in Table 4 are possible to replicate.

For running this example is needed install TensorFlow 2.x Framework and Python3.
Install tensorflow:
```
pip3 install tensorflow
```

Then run example

```
./main.py
```

for compute this table:


|n  |     Network - Code - Decision |      Accuracy|        FCA    |         Undetectable Error   |   Reliability|
|---|-------------------------------|--------------|---------------|------------------------------|---------------|
| 1|            CNN1 - CRC7 - ZADEH:|      92.44   |        92.01  |                 6.77         |           93.15|
| 2|         CNN1 - ONE_HOT - ZADEH:|      92.01   |        91.56  |                 7.29         |           92.63|
| 3|       CNN1 - ONE_HOT - SOFTMAX:|      92.15   |        92.15  |                 7.85         |           92.15|
| 4|          CNN1 - CRC7 - SOFTMAX:|      91.85   |        91.85  |                 8.15         |           91.85|
| 5|    CNN1 - CRC7 - BIT_THRESHOLD:|      89.56   |        89.56  |                 6.80         |           92.94|
| 6|   CNN1 - CRC_HADAMARD - EUCLID:|      89.92   |        88.60  |                 8.20         |           91.53|
| 7|            CNN2 - CRC7 - ZADEH:|      93.97   |        93.47  |                 5.44         |           94.50|
| 8|         CNN2 - ONE_HOT - ZADEH:|      93.40   |        92.66  |                 5.76         |           94.15|
| 9|       CNN2 - ONE_HOT - SOFTMAX:|      94.07   |        94.07  |                 5.93         |           94.07|
|10|          CNN2 - CRC7 - SOFTMAX:|      93.88   |        93.88  |                 6.12         |           93.88|
|11|      ResNet20v2 - CRC7 - ZADEH:|      91.56   |        91.27  |                 7.97         |           91.97|
|12|   ResNet20v2 - ONE_HOT - ZADEH:|      84.16   |        83.28  |                 14.50        |           85.17|
|13| ResNet20v2 - ONE_HOT - SOFTMAX:|      91.67   |        91.67  |                 8.33         |           91.67|
|14|    ResNet20v2 - CRC7 - SOFTMAX:|      91.34   |        91.34  |                 8.66         |           91.34|

