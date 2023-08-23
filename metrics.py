import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns


def rr(actual: list, predicted: list, k=3) -> float:
    """
    Reciprocal Rank: Relevant Item이 추천 리스트 상 몇 번째로 빨리 나타나는가
    추천 상의 몇 개의 relevant item이 등장하든, 첫 번째로 나오는 relevant item만 고려함
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    hit_idxs = []
    for t in actual:
        if t in predicted:
            hit_idxs.append(predicted.index(t) + 1)  # relevant item 위치 추가

    if hit_idxs:  # 맞춘 게 있다면
        first_idx = np.min(hit_idxs)
        score = 1.0 / first_idx
    else:
        score = 0.0
    return score


def mrr(true_purchases: dict, recommendations: dict, k=3):
    """
    Mean Reciprocal Rank at K
    : 전체 유저에 대한 Relevant Item이 추천 리스트 상 몇 번째로 빨리 나타나는가를 측정
      추천 상의 몇 개의 relevant item이 등장하든, 첫 번째로 나오는 relevant item만 고려함
    """
    rr_list = []
    for (cust1, true), (cust2, rec) in zip(true_purchases.items(), recommendations.items()):
        assert cust1 == cust2

        rr_list.append(rr(true, rec, k))

    mrr = np.mean(rr_list)
    return mrr


def _precision(predicted, actual) -> float:
    # ref: # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L130
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec


def apk(actual: list, predicted: list, k=10) -> float:
    if not predicted or not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def ark(actual: list, predicted: list, k=10) -> float:
    # ref: # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L130
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def hr(actual: list, predicted: list, k=10) -> float:
    """
    HIT = 1 if hits in test else 0
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0.0
    for i, a in enumerate(actual):
        if a in predicted:
            num_hits = 1.0

    return num_hits


def mapk(true_purchases: dict, recommendations: dict, k=3):
    """
    mean average precision at k
    : 추천 리스트 중 실제 사용자가 구매한 상품의 비율 (순서 고려)
    """
    apk_list = []
    for (cust1, true), (cust2, rec) in zip(true_purchases.items(), recommendations.items()):
        assert cust1 == cust2

        apk_list.append(apk(true, rec, k))

    mapk = np.mean(apk_list)
    return mapk


def mark(true_purchases: dict, recommendations: dict, k=3):
    """
    mean average recall at k
    : 실제 사용자가 구매한 상품 중 추천 리스트가 맞춘 비율 (순서 고려)
    """
    ark_list = []
    for (cust1, true), (cust2, rec) in zip(true_purchases.items(), recommendations.items()):
        assert cust1 == cust2

        ark_list.append(ark(true, rec, k))

    return np.mean(ark_list)


def hrk(true_purchases: dict, recommendations: dict, k=3):
    """
    Average Hit Ratio at k
    : 추천 리스트 k개 중 상품을 구매한 비율 (순서고려 x)
    : 예를 들어 3개 추천해서 2개 맞으면 0.66667
    """
    hit_list = []
    for (cust1, true), (cust2, rec) in zip(true_purchases.items(), recommendations.items()):
        assert cust1 == cust2

        hit_list.append(hr(true, rec, k))
    return np.mean(hit_list)


def user_coverage(product_table, train_df):
    date = pd.to_datetime(product_table['Fetch_date']).unique()
    last_month = str(date.max().date())

    no_recommend = set(product_table['Customer_id']) - set(train_df['Customer_id'])

    tmp = product_table[product_table['Fetch_date'] != last_month]
    target_cust = tmp['Customer_id'].nunique()
    no_rec_cust = tmp[tmp['Customer_id'].isin(no_recommend)]
    no_rec_cust = no_rec_cust['Customer_id'].nunique()
    user_coverage = (target_cust - no_rec_cust) / target_cust
    return target_cust, no_rec_cust, user_coverage


def model_comparision(models: list, k_max: int = 5) -> pd.DataFrame:
    model_cnt = len(models)
    model_names = [model.__name__ for model in models]

    hrks = []
    mapks = []
    mrrs = []
    coverages = []
    marks = []

    for k in range(1, 6):
        for model in models:
            metrics_dict = model.metrics_at_k(k)
            hrks.append(metrics_dict['hrk'])
            mapks.append(metrics_dict['mapk'])
            marks.append(metrics_dict['mark'])
            mrrs.append(metrics_dict['mrr'])
            coverages.append(metrics_dict['item_coverage'])

    result_df = pd.DataFrame(data=[coverages, mrrs, hrks, mapks, marks],
                             index=['COVERAGE@K', 'MRR@K', 'HR@K', 'MAP@K', 'MAR@K']).T
    result_df['model'] = model_names * 5
    result_df['k'] = np.repeat(range(1, k_max + 1), model_cnt)
    return result_df


def comparison_plot_bar(result_df: pd.DataFrame, metric='MAP@K', filepath=None) -> None:
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,6))
    sns.barplot(data=result_df, y='k', x=metric, hue='model', orient='h')
    plt.title(f"Model Comparison: {metric}", fontsize=15)

    if filepath is not None:
        plt.savefig(filepath)
    plt.show()
    return None





if __name__ == '__main__':
    trues = OrderedDict({0: ['A', 'B', 'C', 'D'],
                         1: ['A', 'B', 'C', 'D'],
                         2: ['W', 'Y']})

    recs = OrderedDict({0: ['F', 'B', 'A'],
                        1: ['Z', 'X', 'C'],
                        2: ['A', 'B', 'C']})

    # 순서를 고려하지 않는 metrics
    print("RR", rr(trues[0], recs[0]))  # RR: 가장 먼저 나온 relevant item의 역수
    print("MRR", mrr(trues, recs))   # MRR: Mean of RR -> 1/3*(1/2 + 1/3 + 0)

    print("Hit Count", hr(trues[0], recs[0], k=2))    # Hit: 추천 상품 리스트 중 실제 구매한 상품이 있는 경우 1, 아니면 0
    print("HR@K", hrk(trues, recs, k=2))   # Hit Ratio: K개 까지 추천했을 때, 전체 유저 중 추천리스트의 상품을 구매한 비율

    print("Precision", _precision(recs[0], trues[0]))   # precision: 예측 중 맞춘 개수의 비율

    # 순서를 고려한 metrics (for implicit)
    print("AP@K", apk(trues[0], recs[0], k=3))   # APK: K개까지 추천했을 때, 순서를 고려한 실제 사용자가 구매한 상품의 비율
    print("MAP@K", mapk(trues, recs, k=3))  # MAPK: Mean of APK

    print("AR@K", ark(trues[0], recs[0], k=3))   # ARK: K개까지 추천했을 때, 순서를 고려한 실제 사용자가 구매한 상품 중 추천된 상품의 비율
    print("MAR@K", mark(trues, recs, k=3))  # MAPK: Mean of APK
