# -*-coding: Utf-8 -*-
# @File : 5.taskClassifier .py
# author: 张英杰
# Time：2024/4/14
import torch
from datasets import load_dataset,load_from_disk


#定义数据集
class Dataset(torch.utils.data.Dataset):
    # def __init__(self, split):
    #     ##原先代码self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)其中包含对关键字的划分，
    #     ##但是我们此处使用了load_from_disk，所以不会存在split的划分
    #     self.dataset = load_from_disk('./data/ChnSentiCorp')
    #
    def __init__(self, split):
        # 加载整个数据集字典
        self.dataset_dict = load_from_disk('./data/ChnSentiCorp')

        # 根据split参数选择特定的数据集分割
        if split in self.dataset_dict:
            self.dataset = self.dataset_dict[split]
        else:
            raise ValueError(f"Invalid split: {split}. Available splits are: {list(self.dataset_dict.keys())}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


dataset = Dataset('train')
print(Dataset)
print(dataset)
print(len(dataset))  ##如果出现长度为3的输出，可能的原因是没有对dataset进行拆分
# print(len(dataset), dataset[0])
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


from transformers import BertTokenizer

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

print(token)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print(len(loader))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)

print(out.last_hidden_state.shape)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out


model = Model()

model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
from transformers import AdamW

#训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print(i, loss.item(), accuracy)

    if i == 300:
        break



print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#测试
def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):

        if i == 5:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)


test()