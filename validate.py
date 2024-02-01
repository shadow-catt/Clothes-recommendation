# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k] # 截断12个预测值

    score = 0.0 
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]: # p not in predicted[:i] 意味着 pred不能重复出现
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0 # gt为空，返回0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


transactions = pd.read_pickle("/dataset/transactions_train.pkl") # 读取 transactions_train
dataset_oof = pickle.load("/dataset/valid_all_100.pkl")

pred = dataset_oof[['user', 'item']].reset_index(drop=True) # 验证集的 user-item对
pred['pred'] = model.predict(dataset_oof[feature_columns]) # 验证集的 模型输出值

pred = pred.groupby(['user', 'item'])['pred'].max().reset_index() # user-item对 的模型输出值 的最大值
pred = pred.sort_values(by=['user', 'pred'], ascending=False).reset_index(drop=True).groupby('user')['item'].apply(lambda x: list(x)[:12]).reset_index() # 获取每个user 预测概率最大的前12个item

gt = transactions.query("week == 0").groupby('user')['item'].apply(list).reset_index().rename(columns={'item': 'gt'}) # 获取groud truth
merged = gt.merge(pred, on='user', how='left') 
merged['item'] = merged['item'].fillna('').apply(list) # FN 填充为 ""

merged.to_pickle(f'{output_dir}/merged_100.pkl') # 保存gt和预测结果
# dataset_oof.to_pickle(f'{output_dir}/valid_all_100.pkl') # 保存验证集


print('mAP@12:', mapk(merged['gt'], merged['item'])) # 计算验证集score