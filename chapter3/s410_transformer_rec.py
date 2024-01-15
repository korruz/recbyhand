import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
from chapter3 import s14_RNN_data_prepare as dp
from torch.utils.data import DataLoader
import torch
from torch import nn
from data_set import filepaths as fp
from tqdm import tqdm
from chapter3 import s49_transformer as TE


class Transformer4Rec(nn.Module):
    def __init__(
        self, n_items, all_seq_lens, e_dim=128, n_heads=3, n_layers=2, alpha=0.2
    ):
        """
        :param n_items: 总物品数量
        :param all_seq_lens: 序列总长度，包含历史物品序列及目标物品
        :param e_dim: 向量维度
        :param n_heads: Transformer中多头注意力层的头目数
        :param n_layers: Transformer中的encoder_layer层数
        :param alpha: 辅助损失函数的计算权重
        """
        super(Transformer4Rec, self).__init__()
        self.items = nn.Embedding(n_items, e_dim, max_norm=1)
        self.encoder = TE.TransformerEncoder(e_dim, e_dim // 2, n_heads, n_layers)
        self.mlp = self.__MLP(e_dim * all_seq_lens)
        self.BCEloss = nn.BCELoss()

        self.decoder = TE.TransformerDecoder(e_dim, e_dim // 2, n_heads, n_layers)
        self.auxDense = self.__Dense4Aux(e_dim, n_items)
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.n_items = n_items

    def __MLP(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 2, dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def __Dense4Aux(self, dim, n_items):
        return nn.Sequential(nn.Linear(dim, n_items), nn.Softmax())

    # 历史物品序列预测的前向传播
    def forwardPredHistory(self, outs, history_seqs):
        # outs:[batch_size, len_seqs, e_dim]
        # [batch_size, len_seqs, e_dim]
        history_seqs_embds = self.items(history_seqs)
        # [batch_size, len_seqs, e_dim]
        outs = self.decoder(history_seqs_embds, outs)
        # [batch_size, len_seqs, n_items]
        outs = self.auxDense(outs)
        # [batch_size*len_seqs, n_items]
        outs = outs.reshape(-1, self.n_items)
        # [batch_size*len_seqs]
        history_seqs = history_seqs.reshape(-1)
        return self.alpha * self.crossEntropyLoss(outs, history_seqs)

    # 推荐预测的前向传播
    def forwardRec(self, item_embs, target_item, target_label):
        logit = self.__getReclogit(item_embs, target_item)
        return self.BCEloss(logit, target_label)

    def __getReclogit(self, item_embs, target_item):
        # [ batch_size, 1, dim ]
        one_item = torch.unsqueeze(self.items(target_item), dim=1)
        # [ batch_size, all_seqs_len, dim ]
        all_item_embs = torch.cat([item_embs, one_item], dim=1)
        # [ batch_size, all_seqs_len * dim ]
        all_item_embs = torch.flatten(all_item_embs, start_dim=1)
        # [ batch_size, 1 ]
        logit = self.mlp(all_item_embs)
        # [ batch_size ]
        logit = torch.squeeze(logit)
        return logit

    def forward(self, x, history_seqs, target_item, target_label):
        # [ batch_size, seqs_len, dim ]
        item_embs = self.items(x)
        # [ batch_size, seqs_len, dim ]
        item_embs = self.encoder(item_embs)
        recLoss = self.forwardRec(item_embs, target_item, target_label)
        auxLoss = self.forwardPredHistory(item_embs, history_seqs)
        return recLoss + auxLoss

    def predict(self, x, target_item):
        item_embs = self.items(x)
        item_embs = self.encoder(item_embs)
        return self.__getReclogit(item_embs, target_item)


# 做评估
def doEva(net, test_triple):
    d = torch.LongTensor(test_triple)
    x = d[:, :-2]
    item = d[:, -2]
    y = torch.FloatTensor(d[:, -1].detach().numpy())
    with torch.no_grad():
        out = net.predict(x, item)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    return precision, recall, acc


def train(epochs=10, batchSize=1024, lr=0.001, dim=128, eva_per_epochs=1):
    # 读取数据
    train, test, allItems = dp.getTrainAndTestSeqs(fp.Ml_latest_small.SEQS)

    all_seq_lens = len(train[0]) - 1
    # 初始化模型
    net = Transformer4Rec(max(allItems) + 1, all_seq_lens, dim)
    # 初始化优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-3)
    # 开始训练
    for e in range(epochs):
        all_lose = 0
        for seq in tqdm(DataLoader(train, batch_size=batchSize, shuffle=True)):
            x = torch.LongTensor(seq[:, :-2].detach().numpy())
            history_seqs = torch.LongTensor(seq[:, 1:-1].detach().numpy())
            target_item = torch.LongTensor(seq[:, -2].detach().numpy())
            target_label = torch.FloatTensor(seq[:, -1].detach().numpy())

            optimizer.zero_grad()
            loss = net(x, history_seqs, target_item, target_label)
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
