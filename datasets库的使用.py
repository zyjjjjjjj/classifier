# -*-coding: Utf-8 -*-
# @File : datasets库的使用 .py
# author: 张英杰
# Time：2024/4/13
##load_dataset 是 Hugging Face's datasets 库中的一个函数，它允许你加载许多流行的数据集，也允许你加载本地数据集或自定义数据集。
# 以下是如何使用 load_dataset 来加载数据集的基本步骤：
from datasets import load_dataset

# 加载一个预定义的数据集，比如 'squad'
dataset = load_dataset('squad')

# 查看数据集的各个分割
print(dataset)

# 查看训练集的样例
for i in range(5):  # 显示前5个样例
    print(dataset['train'][i])


##加载本地自定义的数据集
##如果你有一个本地数据集或自定义数据集，你需要提供数据集的路径。数据集通常以 CSV、JSON、TXT 或其他格式存储。
##例如，假设你有一个名为 my_dataset.csv 的 CSV 文件，并且它位于当前工作目录中。你可以使用以下代码加载它：


from datasets import load_dataset

# 加载本地CSV数据集
dataset = load_dataset('csv', data_files={'train': 'my_dataset.csv'})

# 查看数据集的样例
for i in range(5):  # 显示前5个样例
    print(dataset['train'][i])