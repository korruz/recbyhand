import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from chapter3 import s14_RNN_data_prepare as dp
from torch.utils.data import DataLoader
import torch
from torch import nn
from data_set import filepaths as fp
from tqdm import tqdm
from torch.nn import Parameter, init


class AUGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(AUGRU, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # 初始化AUGRU单元
        self.augru_cell = AUGRU_Cell(in_dim, hidden_dim)

    def forward(self, x, item):
        """
        :param x: 输入的序列向量，维度为 [ batch_size, seq_lens, dim ]
        :param item: 目标物品的向量
        :return: outs: 所有AUGRU单元输出的隐藏向量[ batch_size, seq_lens, dim ]
                 h: 最后一个AUGRU单元输出的隐藏向量[ batch_size, dim ]
        """
        outs = []
        h = None
        # 开始循环，x.shape[1]是序列的长度
        for i in range(x.shape[1]):
            if h == None:
                # 初始化第一层的输入h
                h = init.xavier_uniform_(
                    Parameter(torch.empty(x.shape[0], self.hidden_dim))
                )
            h = self.augru_cell(x[:, i], h, item)
            outs.append(torch.unsqueeze(h, dim=1))
        outs = torch.cat(outs, dim=1)
        return outs, h


# AUGRU单元
class AUGRU_Cell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """
        :param in_dim: 输入向量的维度
        :param hidden_dim: 输出的隐藏层维度
        """
        super(AUGRU_Cell, self).__init__()

        # 初始化更新门的模型参数
        self.Wu = init.xavier_uniform_(Parameter(torch.empty(in_dim, hidden_dim)))
        self.Uu = init.xavier_uniform_(Parameter(torch.empty(in_dim, hidden_dim)))
        self.bu = init.xavier_uniform_(Parameter(torch.empty(1, hidden_dim)))

        # 初始化重置门的模型参数
        self.Wr = init.xavier_uniform_(Parameter(torch.empty(in_dim, hidden_dim)))
        self.Ur = init.xavier_uniform_(Parameter(torch.empty(in_dim, hidden_dim)))
        self.br = init.xavier_uniform_(Parameter(torch.empty(1, hidden_dim)))

        # 初始化计算h~的模型参数
        self.Wh = init.xavier_uniform_(Parameter(torch.empty(hidden_dim, hidden_dim)))
        self.Uh = init.xavier_uniform_(Parameter(torch.empty(hidden_dim, hidden_dim)))
        self.bh = init.xavier_uniform_(Parameter(torch.empty(1, hidden_dim)))

        # 初始化注意计算里的模型参数
        self.Wa = init.xavier_uniform_(Parameter(torch.empty(hidden_dim, in_dim)))

    # 注意力的计算
    def attention(self, x, item):
        """
        :param x: 输入的序列中第t个向量 [ batch_size, dim ]
        :param item: 目标物品的向量 [ batch_size, dim ]
        :return: 注意力权重 [ batch_size, 1 ]
        """
        hW = torch.matmul(x, self.Wa)
        hWi = torch.sum(hW * item, dim=1)
        hWi = torch.unsqueeze(hWi, 1)
        return torch.softmax(hWi, dim=1)

    def forward(self, x, h_1, item):
        """
        :param x:  输入的序列中第t个物品向量 [ batch_size, in_dim ]
        :param h_1:  上一个AUGRU单元输出的隐藏向量 [ batch_size, hidden_dim ]
        :param item: 目标物品的向量 [ batch_size, in_dim ]
        :return: h 当前层输出的隐藏向量 [ batch_size, hidden_dim ]
        """
        # [ batch_size, hidden_dim ]
        u = torch.sigmoid(
            torch.matmul(x, self.Wu) + torch.matmul(h_1, self.Uu) + self.bu
        )
        # [ batch_size, hidden_dim ]
        r = torch.sigmoid(
            torch.matmul(x, self.Wr) + torch.matmul(h_1, self.Ur) + self.br
        )
        # [ batch_size, hidden_dim ]
        h_hat = torch.tanh(
            torch.matmul(x, self.Wh) + r * torch.matmul(h_1, self.Uh) + self.bh
        )
        # [ batch_size, 1 ]
        a = self.attention(x, item)
        # [ batch_size, hidden_dim ]
        u_hat = a * u
        # [ batch_size, hidden_dim ]
        h = (1 - u_hat) * h_1 + u_hat * h_hat
        # [ batch_size, hidden_dim ]
        return h


# Dice激活单元
class Dice(nn.Module):
    def __init__(self, a=0.1):
        super(Dice, self).__init__()
        self.a = a

    def forward(self, embs):
        BN = nn.BatchNorm1d(embs.shape[1])
        prob = torch.sigmoid(BN(embs))
        return prob * embs + (1 - prob) * self.a * embs


class DIEN(nn.Module):
    def __init__(self, n_items, dim=128, alpha=0.2):
        super(DIEN, self).__init__()
        self.dim = dim
        self.alpha = alpha  # 计算辅助损失函数时的权重
        self.n_items = n_items
        self.BCELoss = nn.BCELoss()

        # 随机初始化所有特征的特征向量
        self.items = nn.Embedding(n_items, dim, max_norm=1)

        # 初始化兴趣抽取层的GRU网络，直接用pyTorch里现成的即可
        self.GRU = nn.GRU(dim, dim, batch_first=True)
        # 初始化兴趣演化层的AUGRU网络，因无现成模型，所以使用我们自己编写的AUGRU
        self.AUGRU = AUGRU(dim, dim)

        # 初始化最终ctr预测的mlp网络, 激活函数采用Dice
        self.dense1 = self.dense_layer(dim * 2, dim, Dice)
        self.dense2 = self.dense_layer(dim, dim // 2, Dice)
        self.f_dense = self.dense_layer(dim // 2, 1, nn.Sigmoid)

    # 全连接层
    def dense_layer(self, in_features, out_features, act):
        return nn.Sequential(nn.Linear(in_features, out_features), act())

    # 辅助损失函数的计算过程
    def forwardAuxiliary(self, outs, item_embs, history_labels):
        """
        :param item_embs: 历史序列物品的向量 [ batch_size, len_seqs, dim ]
        :param outs: 兴趣抽取层GRU网络输出的outs [ batch_size, len_seqs, dim ]
        :param history_labels: 历史序列物品标注 [ batch_size, len_seqs, 1 ]
        :return: 辅助损失函数
        """
        # [ batch_size * len_seqs, dim ]
        item_embs = item_embs.reshape(-1, self.dim)
        # [ batch_size * len_seqs, dim ]
        outs = outs.reshape(-1, self.dim)
        # [ batch_size * len_seqs ]
        out = torch.sum(outs * item_embs, dim=1)
        # [ batch_size * len_seqs, 1 ]
        out = torch.unsqueeze(torch.sigmoid(out), 1)
        # [ batch_size * len_seqs,1 ]
        history_labels = history_labels.reshape(-1, 1).float()
        return self.BCELoss(out, history_labels)

    def __getRecLogit(self, h, item):
        # 将AUGRU输出的h向量与目标物品相片拼接,之后经MLP传播
        concatEmbs = torch.cat([h, item], dim=1)
        logit = self.dense1(concatEmbs)
        logit = self.dense2(logit)
        logit = self.f_dense(logit)
        logit = torch.squeeze(logit)
        return logit

    # 推荐CTR预测的前向传播
    def forwardRec(self, h, item, y):
        logit = self.__getRecLogit(h, item)
        y = y.float()
        return self.BCELoss(logit, y)

    # 整体前向传播
    def forward(self, history_seqs, history_labels, target_item, target_label):
        # [ batch_size, len_seqs, dim ]
        item_embs = self.items(history_seqs)

        outs, _ = self.GRU(item_embs)
        # 利用GRU输出的outs得到辅助损失函数
        auxi_loss = self.forwardAuxiliary(outs, item_embs, history_labels)
        # [ batch_size, dim]
        target_item_embs = self.items(target_item)

        # 利用GRU输出的outs与目标目标的向量输入进兴趣演化层的AUGRU网络, 得到最后一层的输出h
        _, h = self.AUGRU(outs, target_item_embs)

        # 得到CTR预估的损失函数
        rec_loss = self.forwardRec(h, target_item_embs, target_label)

        # 将辅助损失函数与CTR预估损失函数加权求和输出
        return self.alpha * auxi_loss + rec_loss

    # 因为模型forward函数输出的是损失函数值，所以另起一个预测函数方便预测及评估
    def predict(self, x, item):
        item_embs = self.items(x)
        outs, _ = self.GRU(item_embs)
        one_item = self.items(item)
        _, h = self.AUGRU(outs, one_item)
        logit = self.__getRecLogit(h, one_item)
        return logit


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    history_seqs = d[:, 0, :-1]
    target_item = d[:, 0, -1]
    y = d[:, 1, -1].float()

    with torch.no_grad():
        out = net.predict(history_seqs, target_item)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return precision, recall, acc


def train(epochs=10, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train, test, allItems = dp.getTrainAndTestSeqsWithNeg(fp.Ml_latest_small.SEQS_NEG)
    # 初始化模型
    net = DIEN(max(allItems) + 1, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in tqdm(
            DataLoader(
                train,
                batch_size=batchSize,
                shuffle=True,
            )
        ):
            history_seqs = torch.LongTensor(seq[:, 0, :-1].detach().numpy())
            history_labels = torch.LongTensor(seq[:, 1, :-1].detach().numpy())
            target_item = torch.LongTensor(seq[:, 0, -1].detach().numpy())
            target_label = torch.LongTensor(seq[:, 1, -1].detach().numpy())
            optimizer.zero_grad()
            loss = net(history_seqs, history_labels, target_item, target_label)
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
