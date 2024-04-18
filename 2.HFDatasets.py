# -*-coding: Utf-8 -*-
# @File : HFDatasets .py
# author: 张英杰
# Time：2024/4/13
# from datasets import load_dataset
#
# #加载数据
# #注意：如果你的网络不允许你执行这段的代码，则直接运行【从磁盘加载数据】即可，我已经给你准备了本地化的数据文件
# #转载自seamew/ChnSentiCorp
# dataset = load_dataset(path='lansinuote/ChnSentiCorp')
#
# dataset
#
# #保存数据集到磁盘
# #注意：运行这段代码要确保【加载数据】运行是正常的，否则直接运行【从磁盘加载数据】即可
# dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

# 运行以上代码失败的原因时hugging face 把自己的数据直接保存在谷歌云盘里，由于网络原因，部分用户可能不能访问谷歌云盘，导致出现了报错问题

from datasets import load_from_disk
dataset = load_from_disk("./data/ChnSentiCorp")

print(dataset) #从本地加载数据完成

#取出训练集
dataset = dataset['train']

print(dataset)

#查看一个数据
print(dataset[0])

#sort

#未排序的label是乱序的
print(dataset['label'][:10])##[:10]是返回列表前十个元素

#排序之后label有序了
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])##也就是表明数据的类型已经变成000000000000.。。。。111111的形式

#shuffle

#打乱顺序
shuffled_dataset = sorted_dataset.shuffle(seed=42)



print(shuffled_dataset['label'][:10])

#select
print(dataset.select([0, 10, 20, 30, 40, 50]))


#filter过滤函数，用于筛选出符合条件的数据
def f(data):
    return data['text'].startswith('选择')


start_with_ar = dataset.filter(f)

print(len(start_with_ar), start_with_ar['text'])##此处过滤器的作用是筛选出具有“选择”两个字开头的text

#train_test_split, 切分训练集和测试集
print(dataset.train_test_split(test_size=0.1))

#shard
#把数据切分到4个桶中,均匀分配
dataset.shard(num_shards=4, index=0)

#rename_column
dataset.rename_column('text', 'textA')

#remove_columns
dataset.remove_columns(['text'])

#map
def f(data):
    data['text'] = 'My sentence: ' + data['text']
    return data


datatset_map = dataset.map(f)

datatset_map['text'][:5]

#set_format
dataset.set_format(type='torch', columns=['label'])

dataset[0]

#第三章/导出为csv格式

from datasets import load_dataset
dataset = load_from_disk('./data/ChnSentiCorp')
print(dataset)
dataset = dataset["train"]
print(dataset)
dataset.to_csv(path_or_buf='./data/ChnSentiCorp2.csv')

#加载csv格式数据
csv_dataset = load_dataset(path='csv',
                           data_files='./data/ChnSentiCorp2.csv',
                           split='train')
print(csv_dataset[20])
print(csv_dataset)

# #第三章/导出为json格式   以下代码需要修正，也就是需要load_from_disk来导入数据
# dataset = load_dataset(path='./data/ChnSentiCorp', split='train')
# dataset.to_json(path_or_buf='./data/ChnSentiCorp2.json')
#
# #加载json格式数据
# json_dataset = load_dataset(path='json',
#                             data_files='./data/ChnSentiCorp2.json',
#                             split='train')
# json_dataset[20]