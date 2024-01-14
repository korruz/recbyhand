from chapter2 import dataloader
from data_set import filepaths as fp
from tqdm import tqdm #进度条的库
from chapter2 import s2_basicSim as b_sim
from utils import evaluate
import collections



#集合形式读取数据, 返回{uid1:{iid1,iid2,iid3}}
def getSet( triples ):
    #用户喜欢的物品集
    user_pos_items = collections.defaultdict( set )
    #用户不喜欢的物品集
    user_neg_items = collections.defaultdict( set )
    #用户交互过的所有物品集
    user_all_items = collections.defaultdict(set)
    #已物品为索引，喜欢物品的用户集
    item_users = collections.defaultdict( set )
    for u, i, r in triples:
        user_all_items[u].add(i)
        if r == 1:
            user_pos_items[u].add(i)
            item_users[i].add(u)
        else:
            user_neg_items[u].add(i)
    return user_pos_items, item_users, user_neg_items, user_all_items

#knn算法
def knn4set(trainset, k, sim_method):
    '''
    :param trainset: 训练集合
    :param k: 近邻数量
    :param sim_method: 相似度方法
    :return: {样本1:[近邻1,近邻2，近邻3]}
    '''
    sims = {}
    #两个for循环遍历训练集合
    for e1 in tqdm(trainset):
        ulist = []#初始化一个列表来记录样本e1的近邻
        for e2 in trainset:
            #如果两个样本相同则跳过
            if e1 == e2 or \
                    len(trainset[e1]&trainset[e2]) == 0:
                #如果两个样本的交集为0也跳过
                continue
            #用相似度方法取得两个样本的相似度
            sim = sim_method(trainset[e1], trainset[e2])
            ulist.append((e2, sim))
        #排序后取前K的样本
        sims[e1] = [i[0] for i in
                    sorted(ulist, key=lambda x:x[1],
                           reverse=True)[:k]]
    return sims

#得到基于相似用户的推荐列表
def get_recomedations_by_usrCF( user_sims, user_o_set ):
    '''
    :param user_sims: 用户的近邻集:{样本1:[近邻1,近邻2，近邻3]}
    :param user_o_set: 用户的原本喜欢的物品集合:{用户1:{物品1,物品2，物品3}}
    :return: 每个用户的推荐列表{用户1:[物品1，物品2，物品3]}
    '''
    recomedations = collections.defaultdict(set)
    for u in user_sims:
        for sim_u in user_sims[u]:
            #将近邻用户喜爱的电影与自己观看过的电影去重后推荐给自己
            recomedations[u] |= (user_o_set[sim_u] - user_o_set[u])
    return recomedations

#得到基于相似物品的推荐列表
def get_recomedations_by_itemCF( item_sims, user_o_set ):
    '''
    :param item_sims: 物品的近邻集:{样本1:[近邻1,近邻2，近邻3]}
    :param user_o_set: 用户的原本喜欢的物品集合:{用户1:{物品1,物品2，物品3}}
    :return: 每个用户的推荐列表{用户1:[物品1，物品2，物品3]}
    '''
    recomedations = collections.defaultdict(set)
    for u in user_o_set:
        for item in user_o_set[u]:
            # 将自己喜欢物品的近邻物品与自己观看过的视频去重后推荐给自己
            if item in item_sims:
                recomedations[u] |= set( item_sims[item] ) - user_o_set[u]
    return recomedations

#得到基于UserCF的推荐列表
def trainUserCF( user_items_train, sim_method, user_all_items, k = 5 ):
    user_sims = knn4set( user_items_train, k, sim_method )
    recomedations = get_recomedations_by_usrCF( user_sims, user_all_items )
    return recomedations

#得到基于ItemCF的推荐列表
def trainItemCF( item_users_train, sim_method, user_all_items, k = 5 ):
    item_sims = knn4set( item_users_train, k, sim_method )
    recomedations = get_recomedations_by_itemCF( item_sims, user_all_items )
    return recomedations

def evaluation( test_set, user_neg_items, pred_set ):
    total_r = 0.0
    total_p = 0.0
    has_p_count = 0
    for uid in test_set:
        if len(test_set[uid]) > 0:
            p = evaluate.precision4Set( test_set[uid], user_neg_items[uid], pred_set[uid] )
            if p:
                total_p += p
                has_p_count += 1
            total_r += evaluate.recall4Set( test_set[uid], pred_set[uid] )

    print("Precision {:.4f} | Recall {:.4f}".format(total_p / has_p_count, total_r / len(test_set)))


if __name__ == '__main__':
    _, _, train_set, test_set = dataloader.readRecData(fp.Ml_100K.RATING, test_ratio=0.1)
    user_items_train, item_users_train, _ ,user_all_items= getSet(train_set)

    user_pos_items_test, _, user_neg_items_test,_ = getSet(test_set)

    recomedations_by_userCF = trainUserCF( user_items_train, b_sim.cos4set, user_all_items, k=5 )
    recomedations_by_itemCF = trainItemCF( item_users_train, b_sim.cos4set, user_all_items, k=5 )

    print(user_pos_items_test)
    print('user_CF')
    evaluation( user_pos_items_test, user_neg_items_test , recomedations_by_userCF )
    print('item_CF')
    evaluation( user_pos_items_test, user_neg_items_test , recomedations_by_itemCF )

    '''
    user_CF
    Precision 0.6774 | Recall 0.8186
    item_CF
    Precision 0.7095 | Recall 0.6235
    '''