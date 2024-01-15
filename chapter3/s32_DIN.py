import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from chapter3 import s14_RNN_data_prepare as dp
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
from data_set import filepaths as fp
from torch.nn import Parameter, init


class DIN(nn.Module):
    def __init__(self, n_items, dim=128, t=64):
        super(DIN, self).__init__()
        # 随机初始化所有物品向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)

        self.fliner = nn.Linear(dim * 2, 1)
        # 注意力计算中的线性层
        self.attention_liner = nn.Linear(dim, t)
        self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))

        # 初始化一个BN层，在dice计算时会用到
        self.BN = nn.BatchNorm1d(1)

    # Dice激活函数
    def Dice(self, embs, a=0.1):
        prob = torch.sigmoid(self.BN(embs))
        return prob * embs + (1 - prob) * a * embs

    # 注意力计算
    def attention(self, embs):
        # embs: [ batch_size, k ]
        # [ batch_size, t ]
        embs = self.attention_liner(embs)
        # [ batch_size, t ]
        embs = torch.relu(embs)
        # [ batch_size, 1 ]
        embs = torch.matmul(embs, self.h)
        # [ batch_size, 1 ]
        atts = torch.softmax(embs, dim=1)
        return atts

    def forward(self, x, item, isTrain=True):
        # [ batch_size, len_seqs, dim ]
        item_embs = self.items(x)
        # [ batch_size, len_seqs, 1 ]
        atts = self.attention(item_embs)
        # [ batch_size, dim]
        sumWeighted = torch.sum(item_embs * atts, dim=1)
        # [ batch_size, dim]
        one_item = self.items(item)
        # [ batch_size, dim*2 ]
        out = torch.cat([sumWeighted, one_item], dim=1)
        # 训练时采取dropout来防止过拟合
        if isTrain:
            out = F.dropout(out)
        # [ batch_size, 1 ]
        out = self.fliner(out)
        out = self.Dice(out)
        # [ batch_size ]
        out = torch.squeeze(out)
        logit = torch.sigmoid(out)
        return logit


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = torch.FloatTensor(d[:, -1].detach().numpy())
    with torch.no_grad():
        out = net(x, item)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return precision, recall, acc


def train(epochs=10, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train, test, allItems = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)
    # 初始化模型
    net = DIN(max(allItems) + 1, dim)
    # 定义损失函数
    criterion = torch.nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in DataLoader(
            train,
            batch_size=batchSize,
            shuffle=True,
        ):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            item = torch.LongTensor(seq[:, -2].detach().numpy())
            y = torch.FloatTensor(seq[:, -1].detach().numpy())
            optimizer.zero_grad()
            logits = net(x, item)
            loss = criterion(logits, y)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print(
            "epoch {},avg_loss={:.4f}".format(e, all_lose / (len(train) // batchSize))
        )

        # 评估模型
        if e % eva_per_epochs == 0:
            p, r, acc = doEva(net, train)
            print("train:p:{:.4f}, r:{:.4f}, acc:{:.4f}".format(p, r, acc))
            p, r, acc = doEva(net, test)
            print("test:p:{:.4f}, r:{:.4f}, acc:{:.4f}".format(p, r, acc))


if __name__ == "__main__":
    train()
