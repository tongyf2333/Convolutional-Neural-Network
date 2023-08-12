这是我用python手搓的一个卷积神经网络，没有使用pytorch、tenserflow等框架。

使用了numpy库以及cython来提高运行效率。

卷积层池化层和BN层的代码实现参考了cs231n的assignment2。 [链接](https://cs231n.github.io/assignments2023/assignment2/)

[cs231n网站](https://cs231n.github.io/)

当前mnist测试集准确率：98.49%

正在写RNN[已经写完辣](https://github.com/tongyf2333/Long-short-term-memory/tree/master)

update2023.8.13：修复了CNN_optimized.py中test函数的bug
