# -*-coding: Utf-8 -*-
# @File : HuggingFace .py
# author: 张英杰
# Time：2024/4/10
import transformers
from transformers import BertTokenizer

#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='./bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

print(tokenizer)
print(sents)

#编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,
)

print(out)

 ##这是对以上编码结果的一个解码的过程，bert中将每一个字都当作了一个词来进行处理。
 ## 编码结果中cls、sep类似于句子开头以及句子分割的标志。pad是因为长度不够自动补齐的结果
print(tokenizer.decode(out))

print("===以下是一个增强的编码函数===")
#增强的编码函数
out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tf,pt,np,默认为返回list，这三个取值分别对应了tensorflow，pytorch，numpy
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'])

print("===批量编码句子的结果，实质上是采用了tokenizer中的不同函数===")
print("之前两次使用的编码函数是针对与一个或者说是一个句子来进行操作的，但是batch-encode-plus是针对于一组句子来进行操作的")
#批量编码句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1],sents[2]],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=15,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

print(tokenizer.decode(out['input_ids'][0]), tokenizer.decode(out['input_ids'][1]))

print("===以下是批量的成对采取句子来进行编码的结果===")
#批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    add_special_tokens=True,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,

    #可取值tf,pt,np,默认为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

#input_ids 就是编码后的词
#token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
#special_tokens_mask 特殊符号的位置是1,其他位置是0
#attention_mask pad的位置是0,其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)

print(tokenizer.decode(out['input_ids'][0]))

print("===以下是获取字典的操作===")
#获取字典
zidian = tokenizer.get_vocab()

print(type(zidian), len(zidian), '月光' in zidian)

print("===以下是给字典添加新词和新符号的操作===")

#添加新词
tokenizer.add_tokens(new_tokens=['月光', '希望'])

#添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

zidian = tokenizer.get_vocab()

print(type(zidian), len(zidian), zidian['月光'], zidian['[EOS]'])

print("===编码新加的词===")

#编码新添加的词
out = tokenizer.encode(
    text='月光的新希望[EOS]',
    text_pair=None,

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    return_tensors=None,
)

print(out)

print(tokenizer.decode(out))
print("注意到输出结果中，月光和希望已经成为词汇去出现了，而且【eos】这个我们添加的无意义的符号也被识别了出来")