import pandas as pd
import numpy as np


# 信息熵 H(D)
def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return -sum(np.log2(prob1) * prob1)


# 条件信息熵 H(X|Y)
def ent_cond(data, X, Y):
    e1 = data.groupby(Y).apply(lambda x: ent(x[X]))
    p1 = pd.value_counts(data[Y]) / len(data[Y])
    return sum(e1 * p1)


# 信息增益
def gain(data, X, Y):
    HD = ent(data[X])
    return HD - ent_cond(data, X, Y)


# 信息增益比
def gain_rate(data, X, Y):
    HD = ent(data[X])
    UHD = HD - ent_cond(data, X, Y)
    return UHD / HD if HD > 0 else 0


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "性别": ["男", "女", "男", "女", "女", "男", "男", "女", "男", "女"],
            "职业": ["学生", "码农", "教师", "学生", "码农", "学生", "码农", "学生", "码农", "教师"],
            "电影类别": ["喜剧", "科幻", "爱情", "科幻", "科幻", "喜剧", "科幻", "喜剧", "科幻", "爱情"],
        }
    )

    print(gain_rate(df, "电影类别", "性别"))
    print(gain_rate(df, "电影类别", "职业"))
