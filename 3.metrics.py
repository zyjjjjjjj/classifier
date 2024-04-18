# -*-coding: Utf-8 -*-
# @File : 3.metrics .py
# author: 张英杰
# Time：2024/4/14
from datasets import list_metrics

#列出评价指标
metrics_list = list_metrics()

print(len(metrics_list), metrics_list)

from datasets import load_metric

#加载一个评价指标
metric = load_metric('glue', 'mrpc')

print(metric.inputs_description)

#计算一个评价指标
predictions = [0, 1, 0]
references = [0, 1, 1]

final_score = metric.compute(predictions=predictions, references=references)

print(final_score)