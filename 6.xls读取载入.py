# -*-coding: Utf-8 -*-
# @File : 测试文件 .py
# author: 张英杰
# Time：2024/4/12
import pandas as pd
from datasets import load_dataset
# 读取 CSV 文件
df = pd.read_csv('./data/new.csv', encoding="gbk")  # 如果不知道原始编码，可以尝试不指定编码

# 将 DataFrame 保存为新的 UTF-8 编码的 CSV 文件
df.to_csv('new_file_utf8.csv', index=False, encoding='utf-8-sig')  # 'utf-8-sig' 会添加一个 UTF-8 BOM（字节顺序标记）

#加载csv格式数据
csv_dataset = load_dataset(path='csv',
                           data_files='./new_file_utf8.csv',
                           split='train')
print(csv_dataset)
print(csv_dataset.train_test_split(test_size=0.1))
csv_dataset = csv_dataset.train_test_split(test_size=0.1)


# csv_dataset2 = load_dataset(path='csv',
#                            data_files='./data/new.csv',
#                            split='train')
# print(csv_dataset2)
# print(csv_dataset2.train_test_split(test_size=0.1))  此处代码是错误示范，因为不清楚文件的编码格式，导致出现了bug
