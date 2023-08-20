import numpy as np
from collections import OrderedDict


def _rr(actual: list, predicted: list, k=3) -> float:
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

        rr_list.append(_rr(true, rec, k))

    mrr = np.mean(rr_list)
    return mrr


def _precision(predicted, actual) -> float:
    # ref: # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L130
    prec = [value for value in predicted if value in actual]
    prec = float(len(prec)) / float(len(predicted))
    return prec


def _apk(actual: list, predicted: list, k=10) -> float:
    # ref: # https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py#L130
    if not predicted or not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    true_positives = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            max_ix = min(i + 1, len(predicted))
            score += _precision(predicted[:max_ix], actual)
            true_positives += 1

    if score == 0.0:
        return 0.0

    return score / true_positives


def _ark(actual: list, predicted: list, k=10) -> float:
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


def _hr(actual: list, predicted: list, k=10) -> float:
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

        apk_list.append(_apk(true, rec, k))

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

        ark_list.append(_ark(true, rec, k))

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

        hit_list.append(_hr(true, rec, k))
    return np.mean(hit_list)


if __name__ == '__main__':
    trues = OrderedDict({0: ['A', 'B', 'C', 'D'],
                         1: ['A', 'B', 'C', 'D'],
                         2: ['W', 'Y']})

    recs = OrderedDict({0: ['F', 'B', 'A'],
                        1: ['Z', 'X', 'C'],
                        2: ['A', 'B', 'C']})

    # 순서를 고려하지 않는 metrics
    print("RR", _rr(trues[0], recs[0]))  # RR: 가장 먼저 나온 relevant item의 역수
    print("MRR", mrr(trues, recs))   # MRR: Mean of RR -> 1/3*(1/2 + 1/3 + 0)

    print("Hit Count", _hr(trues[0], recs[0], k=2))    # Hit: 추천 상품 리스트 중 실제 구매한 상품이 있는 경우 1, 아니면 0
    print("HR@K", hrk(trues, recs, k=2))   # Hit Ratio: K개 까지 추천했을 때, 전체 유저 중 추천리스트의 상품을 구매한 비율

    print("Precision", _precision(recs[0], trues[0]))   # precision: 예측 중 맞춘 개수의 비율

    # 순서를 고려한 metrics (for implicit)
    print("AP@K", _apk(trues[0], recs[0], k=3))   # APK: K개까지 추천했을 때, 순서를 고려한 실제 사용자가 구매한 상품의 비율
    print("MAP@K", mapk(trues, recs, k=3))  # MAPK: Mean of APK

    print("AR@K", _ark(trues[0], recs[0], k=3))   # ARK: K개까지 추천했을 때, 순서를 고려한 실제 사용자가 구매한 상품 중 추천된 상품의 비율
    print("MAR@K", mark(trues, recs, k=3))  # MAPK: Mean of APK




