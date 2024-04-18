# -*-coding: Utf-8 -*-
# @File : 测试文件 .py
# author: 张英杰
# Time：2024/4/17
from datasets import load_dataset

# 如果CSV文件在本地，使用以下方式加载
dataset = load_dataset('csv', data_files={'train': './new_file_utf8.csv', 'test': './new_file_utf8.csv', "valid": "./new_file_utf8.csv"})
dataset.save_to_disk("./selfdata")
# 如果CSV文件在Hugging Face datasets仓库中，可以使用对应的ID来加载
# dataset = load_dataset('your_dataset_id')