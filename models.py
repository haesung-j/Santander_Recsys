import warnings
import pandas as pd
import numpy as np
import multiprocessing
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from implicit.evaluation import ranking_metrics_at_k
from collections import OrderedDict
from typing import Tuple

from utils import *
from metrics import *


class RecsysBase:
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None) -> None:
        print("Fitting the model ...")
        return self._fit(X)

    def predict(self, X: pd.DataFrame) -> Tuple[dict, dict]:
        print("Predict ...")
        return self._predict(X)

    def metrics_at_k(self, k: int = 3) -> dict:
        print("Evaluate ...")
        mrr = self._mrr(k)
        mapk = self._mapk(k)
        mark = self._mark(k)
        hrk = self._hrk(k)
        auc, ndcg = self._auc_ndcg(self._target_metric_df, k)
        results = {'mrr': mrr,
                   'mapk': mapk,
                   'mark': mark,
                   'hrk': hrk,
                   'ndcg': ndcg,
                   'auc': auc}

        return results

    def _mrr(self, k: int = 3):
        score = mrr(self.true_purchases, self.recommendations, k)
        return score

    def _mapk(self, k: int = 3):
        score = mapk(self.true_purchases, self.recommendations, k)
        return score

    def _mark(self, k: int = 3):
        score = mark(self.true_purchases, self.recommendations, k)
        return score

    def _hrk(self, k: int = 3):
        score = hrk(self.true_purchases, self.recommendations, k)
        return score

    def _auc_ndcg(self, X: pd.DataFrame, k:int = 3):
        """
        X: stacked 형태의 dataframe
        """
        # 1. train set에 존재하는 customer만 추출
        test_sparse_df = self.sparse_matrix_df(X)
        intersection_customer = set(self.sparse_matrix.index).intersection(set(X['Customer_id']))
        test_sparse_df = test_sparse_df.loc[test_sparse_df.index.isin(intersection_customer)]

        # 2. train set과 동일한 컬럼을 갖도록 sprase maxtrix 수정
        add_cols = [col for col in self.sparse_matrix.columns if col not in test_sparse_df.columns]
        for col in add_cols:
            test_sparse_df[col] = 0.0
        test_sparse_df = test_sparse_df[self.sparse_matrix.columns]  # column(product_id) 순서 일치

        # 3. train set과 동일한 shape의 user-item matrix 생성
        csr_test = np.zeros_like(self.sparse_matrix)
        csr_test = pd.DataFrame(csr_test, index=self.sparse_matrix.index, columns=self.sparse_matrix.columns)

        # 4. user-item matrix 내 test set 값 채워넣기
        for i in tqdm(range(len(test_sparse_df))):
            idx = test_sparse_df.index[i]
            csr_test.loc[idx] = test_sparse_df.loc[idx]

        # 5. csr matrix type으로 변경
        csr_test = csr_matrix(csr_test)

        # 6. implicit.evaluation 내 ranking_metrics_at_k 메서드로 auc 계산
        metrics = ranking_metrics_at_k(model=self.model, train_user_items=self.csr, test_user_items=csr_test, K=k,
                                       num_threads=multiprocessing.cpu_count()-1)
        precision, map, ndcg, auc = metrics['precision'], metrics['map'], metrics['ndcg'], metrics['auc']
        return auc, ndcg

    @staticmethod
    def similarity_df(x: pd.DataFrame, y=None) -> pd.DataFrame:
        if y is None:
            sim_matrix = cosine_similarity(x.values, x.values)
            sim_df = pd.DataFrame(data=sim_matrix, columns=x.index, index=x.index)
        else:
            sim_matrix = cosine_similarity(x.values, y.values)
            sim_df = pd.DataFrame(data=sim_matrix, columns=x.index, index=y.index)

        return sim_df

    @staticmethod
    def sparse_matrix_df(x: pd.DataFrame) -> pd.DataFrame:
        sparse_df = pd.pivot_table(x,
                                   index='Customer_id',
                                   columns='product_id',
                                   values='purchase').fillna(0)
        return sparse_df


# class ItemBasedCF(RecsysBase):
#     def __init__(self, multiprocess=True, n_jobs=-1):
#         super().__init__()
#         # self.item_neighbors = item_neighbors
#         self.multiprocess = multiprocess
#         self.n_jobs = n_jobs
#
#     def _predict(self, X: pd.DataFrame) -> Tuple[dict, dict]:
#         print("Generate Recommendations for all users in test set...")
#         # test_df 와 X(train_df)에 모두 존재하는 Customer_id와 product_id 추출
#         intersection_product_ids = set(self.prediction_result_df.columns).intersection(set(X['product_id'].unique()))
#         intersection_customer_ids = set(self.prediction_result_df.index).intersection(set(X['Customer_id'].unique()))
#         intersection_product_ids = sorted(list(intersection_product_ids))
#         intersection_customer_ids = sorted(list(intersection_customer_ids))
#
#         print(f"train_df와 test_df에 모두 존재하는 product_id 개수: {len(intersection_product_ids):,}개")
#         print(f"train_df와 test_df에 모두 존재하는 Customer_id와 개수: {len(intersection_customer_ids):,}명")
#
#         # train_df와 X(test_df)에 공통으로 존재하는 Customer에 대해서 추천리스트 생성
#         grouped = X.groupby(by='Customer_id')
#         recommendations = OrderedDict()
#         true_purchases = OrderedDict()
#         for customer_id, group in tqdm(grouped):
#             if customer_id in intersection_customer_ids:
#                 already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] == 1].tolist()
#                 preds = self.prediction_result_df.loc[customer_id].sort_values(ascending=False).index
#                 rec_list = preds[~preds.isin(already_have)].values
#                 true_list = group['product_id'].values
#                 recommendations[customer_id] = list(rec_list)
#                 true_purchases[customer_id] = list(true_list)
#
#         self.recommendations = recommendations
#         self.true_purchase = true_purchases
#
#         return recommendations, true_purchases
#
#     def predict_single_user(self, customer_id):
#         train_user_idx = set(self.sparse_matrix.index)
#         # true_list = group['product_id'].values
#         if customer_id not in train_user_idx:
#             print(
#                 f"There is no customer_id: {customer_id} in train set. We cannot make recommendation for this customer.")
#             rec_list = [999]
#             scores = [0]
#         else:
#             already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] == 1].tolist()
#             already_have_names = [prod_id_inv[p] for p in already_have]
#             print(f"{customer_id}가 이미 보유한 상품: {already_have_names}")
#
#             sim_df = self.product_sim_df.loc[already_have]
#             sim_df.index = [prod_id_inv[i] for i in sim_df.index]
#             sim_df.columns = [prod_id_inv[i] for i in sim_df.columns]
#             print(f"보유한 상품과 다른 상품과의 유사도는 다음과 같습니다.\n{sim_df}")
#
#             preds = self.prediction_result_df.loc[customer_id].sort_values(ascending=False)
#             preds = preds[~preds.index.isin(already_have)]
#             scores = preds.values
#             rec_list = preds.index.values
#             print(f"각 상품별 user의 구매 가능 점수는 다음과 같습니다.\n{scores}")
#             print(f"따라서, 다음의 상품을 추천합니다.\n{[prod_id_inv[r] for r in rec_list]}")
#         return rec_list, scores
#
#     def _fit(self, X: pd.DataFrame, y=None) -> None:
#         """
#         1. X(pd.DataFrame)로부터 item-user sparse_matrix를 생성
#         2. sparse_matrix를 기반으로 item-item similarity matrix DataFrame 생성
#         3. 각 user의 구매기록을 바탕으로 item similarty간 가중합을 구한 뒤, user-item feedback 예측
#
#         :param X: pd.DataFrame['Customer_id': int, 'product_id': int, 'purchase': int]
#         :param y: None
#         :return self
#         """
#         # 1. X(pd.DataFrame)로부터 item-user sparse_matrix를 생성
#         self.sparse_matrix = pd.pivot_table(X,
#                                             index='Customer_id',
#                                             columns='product_id',
#                                             values='purchase'
#                                             ).fillna(0)
#
#         # 2. sparse_matrix를 기반으로 item-item similarity matrix DataFrame 생성
#         self.product_sim_df = self.similarity_df(self.sparse_matrix.T)  # 아이템 유사도 행렬
#
#         # 3. 각 user의 구매기록을 바탕으로 item similarty간 가중합을 구한 뒤, user-item feedback 예측
#         customer_grp = X.groupby('Customer_id')  # Customer별로 groupby
#
#         if self.multiprocess:
#             self.prediction_result_df = self._applyparallel(customer_grp, self._find)
#             self.prediction_result_df = self.prediction_result_df.set_index('Customer_id')
#
#         else:
#             self.prediction_result_df = pd.DataFrame(index=customer_grp.indices.keys(),
#                                                      columns=self.product_sim_df.index)
#             for userId, group in tqdm(customer_grp):
#                 product_similarities = self.product_sim_df.loc[
#                     group['product_id']]  # 유저가 구매한 상품에 대한 유사도 행렬(차원: 구매 상품 x 전체 상품)
#                 user_purchase = group['purchase']  # user가 구매한 상품(1) (차원: 구매한 상품 x 1)
#                 sim_sum = product_similarities.sum(axis=0)  # 차원: 전체 상품 x 1
#                 # sim_sum.loc[sim_sum == 0] = 1
#
#                 # userId 전체 purchase predictions -> 유저가 구매한 상품을 기반으로 모든 상품에 대해 유사도 가중합
#                 pred_ratings = np.matmul(product_similarities.T.to_numpy(), user_purchase) / (
#                             sim_sum + 1)  # 유저가 평가한 영화 vs 다른 영화들과의 유사도 %*% 유저의 평가
#                 self.prediction_result_df.loc[userId] = pred_ratings
#
#     def _applyparallel(self, dfgrouped, func):
#         if self.n_jobs == -1:
#             cpu_count = multiprocessing.cpu_count()
#         else:
#             cpu_count = self.n_jobs
#         with Parallel(n_jobs=cpu_count) as parall:
#             res_list = parall(delayed(func)(group) for name, group in tqdm(dfgrouped))
#         return pd.concat(res_list)
#
#     def _find(self, group):
#         product_similarities = self.product_sim_df.loc[
#             group['product_id']]  # customer가 구매한 상품에 대한 유사도 행렬(차원: 구매 상품 x 전체 상품)
#         user_purchase = group['purchase']  # customer가 구매한 상품(1) (차원: 구매한 상품 x 1)
#         sim_sum = product_similarities.sum(axis=0)  # 차원: 전체 상품 x 1
#         # sim_sum.loc[sim_sum == 0] = 1
#
#         # userId 전체 purchase predictions -> customer가 구매한 상품을 기반으로 모든 상품에 대해 유사도 가중합
#         pred_ratings = np.matmul(product_similarities.T.to_numpy(), user_purchase) / (
#                 sim_sum + 1)  # 유저가 구매한 상품 vs 다른 상품들과의 유사도 %*% 유저의 구매여부
#         pred_ratings = pd.DataFrame(pred_ratings).T
#         pred_ratings['Customer_id'] = group['Customer_id'].values[0]
#         return pred_ratings


class ItemBasedCF(RecsysBase):
    def __init__(self, K=23):
        super().__init__()
        self.model = CosineRecommender(K=K)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        # sparse matrix 생성
        self.sparse_matrix = pd.pivot_table(X,
                                            index='Customer_id',
                                            columns='product_id',
                                            values='purchase'
                                            ).fillna(0)
        self.max_items = len(self.sparse_matrix.columns)
        self.csr = csr_matrix(self.sparse_matrix)  # csr matrix 생성
        # inner id 매핑을 위한 dictionary 생성
        self.custid_to_innerid = {u: i for i, u in enumerate(self.sparse_matrix.index.values)}
        self.innerid_to_custid = {v: k for k, v in self.custid_to_innerid.items()}

        self.itemid_to_innerid = {u: i for i, u in enumerate(self.sparse_matrix.columns.values)}
        self.innerid_to_itemid = {v: k for k, v in self.itemid_to_innerid.items()}

        self.model.fit(self.csr)

    def _predict(self, X: pd.DataFrame) -> Tuple[dict, dict]:
        test_user_ids = sorted(list(set(X['Customer_id'])))
        test_inner_user_ids = [self.custid_to_innerid[t] for t in test_user_ids if t in self.custid_to_innerid.keys()]
        ids, scores = self.model.recommend(test_inner_user_ids, self.csr[test_inner_user_ids],
                                           N=20,
                                           filter_already_liked_items=True)
        # 실제 구매 딕셔너리
        recommendations = OrderedDict()
        true_purchases = OrderedDict()
        for i in tqdm(range(len(test_inner_user_ids))):
            p = test_inner_user_ids[i]
            true_purchases[self.innerid_to_custid[p]] = list(
                X[X['Customer_id'] == self.innerid_to_custid[p]]['product_id'].values)
            recommendations[self.innerid_to_custid[p]] = ids[i].tolist()

        self.recommendations = recommendations
        self.true_purchases = true_purchases

        self._target_metric_df = X

        return recommendations, true_purchases

    def explain_single_user(self, customer_id):
        train_customer_idx = set(self.sparse_matrix.index)
        # true_list = group['product_id'].values
        if customer_id not in train_customer_idx:
            print(
                f"There is no customer_id: {customer_id} in train set. We cannot make recommendation for this customer.")
            rec_list = [999]
            scores = [0]
        else:
            inner_user_id = self.custid_to_innerid[customer_id]
            already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] == 1].tolist()
            already_have_names = [v for k, v in prod_id_inv.items() if k in already_have]
            not_have_names = [v for k, v in prod_id_inv.items() if k not in already_have]

            buy_items = [prod_id_inv[v] for v in self._target_metric_df[
                self._target_metric_df['Customer_id'] == customer_id]['product_id'].values]

            # customer가 이미 소유한 상품과 소유하지 않은 상품 간 아이템 유사도 데이터프레임
            sim_item_df = pd.DataFrame(index=not_have_names)
            already_have_len = len(already_have)
            inner_already_have = [self.itemid_to_innerid[k] for k in already_have]

            for i in already_have:
                inner_item_id = self.itemid_to_innerid[i]
                items, sim_scores = self.model.similar_items(inner_item_id, N=self.max_items - already_have_len,
                                                             filter_items=inner_already_have)
                product_ids = [self.innerid_to_itemid[p] for p in items]
                item_names = [prod_id_inv[p] for p in product_ids]
                t = pd.DataFrame(index=item_names)
                t[prod_id_inv[i]] = sim_scores
                sim_item_df = sim_item_df.join(t)

            sim_item_df = sim_item_df.T.fillna(0)

            print(f"{customer_id}가 이미 보유한 상품: {already_have_names}")
            print(f"보유한 상품과 다른 상품과의 유사도는 다음과 같습니다.\n{sim_item_df}")

            item_recs = \
            self.model.recommend(inner_user_id, self.csr[inner_user_id], N=self.max_items - already_have_len,
                                 filter_already_liked_items=True)[0]
            item_scores = \
            self.model.recommend(inner_user_id, self.csr[inner_user_id], N=self.max_items - already_have_len,
                                 filter_already_liked_items=True)[1]

            rec_df = pd.DataFrame(item_scores, item_recs)
            rec_df.index = [prod_id_inv[self.innerid_to_itemid[r]] for r in rec_df.index]

            rec_list = rec_df.index.tolist()
            scores = item_scores
            print(f"각 상품별 user의 구매 가능 점수는 다음과 같습니다.\n{rec_df}")
            print(f"따라서, 다음의 상품을 추천합니다.\n{rec_list}")
            print(f"{customer_id} 고객은 {buy_items}를 구매했습니다.")
        return rec_list, scores


class BPRMF(RecsysBase):
    def __init__(self, factors=100, learning_rate=0.1, regularization=0.1, iterations=100):
        super().__init__()
        self.model = BayesianPersonalizedRanking(factors=factors,
                                                 learning_rate=learning_rate,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 random_state=0)

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self.sparse_matrix = pd.pivot_table(X,
                                            index='Customer_id',
                                            columns='product_id',
                                            values='purchase'
                                            ).fillna(0)
        self.csr = csr_matrix(self.sparse_matrix)  # csr matrix 생성
        # inner id 매핑을 위한 dictionary 생성
        self.custid_to_innerid = {u: i for i, u in enumerate(self.sparse_matrix.index.values)}
        self.innerid_to_custid = {v: k for k, v in self.custid_to_innerid.items()}

        self.itemid_to_innerid = {u: i for i, u in enumerate(self.sparse_matrix.columns.values)}
        self.innerid_to_itemid = {v: k for k, v in self.itemid_to_innerid.items()}

        self.model.fit(self.csr)

    def _predict(self, X: pd.DataFrame) -> Tuple[dict, dict]:
        print("Generate Recommendations for all users in test set...")
        # test_df 와 X(train_df)에 모두 존재하는 Customer_id와 product_id 추출
        intersection_product_ids = set(self.sparse_matrix.columns).intersection(set(X['product_id'].unique()))
        intersection_customer_ids = set(self.sparse_matrix.index).intersection(set(X['Customer_id'].unique()))
        intersection_product_ids = sorted(list(intersection_product_ids))
        intersection_customer_ids = sorted(list(intersection_customer_ids))

        print(f"train_df와 test_df에 모두 존재하는 product_id 개수: {len(intersection_product_ids):,}개")
        print(f"train_df와 test_df에 모두 존재하는 Customer_id와 개수: {len(intersection_customer_ids):,}명")

        # train_df와 X(test_df)에 공통으로 존재하는 Customer에 대해서 추천리스트 생성
        #         grouped = X.groupby(by='Customer_id')
        recommendations = OrderedDict()
        true_purchases = OrderedDict()
        for customer_id in tqdm(intersection_customer_ids):
            inner_user_id = self.custid_to_innerid[customer_id]
            ids, scores = self.model.recommend(inner_user_id, self.csr[inner_user_id], N=24,
                                               filter_already_liked_items=True)
            rec_list = [self.innerid_to_itemid[i] for i in ids]
            true_list = X[X['Customer_id'] == customer_id]['product_id'].values
            recommendations[customer_id] = list(rec_list)
            true_purchases[customer_id] = list(true_list)

        self.recommendations = recommendations
        self.true_purchases = true_purchases

        # predict에 사용한 data를 evaluation에서도 사용하기 위함
        self._target_metric_df = X

        return recommendations, true_purchases

    def explain_single_user(self, customer_id):
        train_customer_idx = set(self.sparse_matrix.index)
        # true_list = group['product_id'].values
        if customer_id not in train_customer_idx:
            print(
                f"There is no customer_id: {customer_id} in train set. We cannot make recommendation for this customer.")
            rec_list = [999]
            scores = [0]
        else:
            max_items = len(self.sparse_matrix.columns)
            inner_user_id = self.custid_to_innerid[customer_id]
            already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] > 0].tolist()
            already_have_names = [v for k, v in prod_id_inv.items() if k in already_have]
            not_have_names = [v for k, v in prod_id_inv.items() if k not in already_have]
            buy_items = [prod_id_inv[v] for v in self._target_metric_df[
                self._target_metric_df['Customer_id'] == customer_id]['product_id'].values]

            # customer가 이미 소유한 상품과 소유하지 않은 상품 간 아이템 유사도 데이터프레임
            sim_item_df = pd.DataFrame(index=not_have_names)
            already_have_len = len(already_have)
            inner_already_have = [self.itemid_to_innerid[k] for k in already_have]

            for i in already_have:
                inner_item_id = self.itemid_to_innerid[i]
                items, sim_scores = self.model.similar_items(inner_item_id, N=max_items - already_have_len,
                                                             filter_items=inner_already_have)
                product_ids = [self.innerid_to_itemid[p] for p in items]
                item_names = [prod_id_inv[p] for p in product_ids]
                t = pd.DataFrame(index=item_names)
                t[prod_id_inv[i]] = sim_scores
                sim_item_df = sim_item_df.join(t)

            sim_item_df = sim_item_df.T.fillna(0)

            print(f"{customer_id}가 이미 보유한 상품: {already_have_names}")
            print(f"보유한 상품과 다른 상품과의 유사도는 다음과 같습니다.\n{sim_item_df}")

            item_recs = self.model.recommend(inner_user_id, self.csr[inner_user_id], N=max_items - already_have_len,
                                             filter_already_liked_items=True)[0]
            item_scores = self.model.recommend(inner_user_id, self.csr[inner_user_id], N=max_items - already_have_len,
                                               filter_already_liked_items=True)[1]

            rec_df = pd.DataFrame(item_scores, item_recs)
            rec_df.index = [prod_id_inv[self.innerid_to_itemid[r]] for r in rec_df.index]

            rec_list = rec_df.index.tolist()
            scores = item_scores
            print(f"각 상품별 user의 구매 가능 점수는 다음과 같습니다.\n{rec_df}")
            print(f"따라서, 다음의 상품을 추천합니다.\n{rec_list}")
            print(f"{customer_id} 고객은 {buy_items}를 구매했습니다.")
        return rec_list, scores

    def _predict_user(self, customer_id):
        inner_user_id = self.custid_to_innerid[customer_id]
        scores = self.model.user_factors[inner_user_id].dot(self.model.item_factors.T)
        return scores





if __name__ == '__main__':
    DATA_PATH = './data'
    product_table = load_product_table(DATA_PATH)
    prod_id, prod_id_inv = get_product_dictionary(product_table)
    train_df = get_implicit_binary(product_table)  # 4월에 고객별 보유중인 상품
    test_df = get_test_set(product_table)  # 5월에 새로 구매한 (고객,상품)

    print(train_df.shape)

    cf = ItemBasedCF(K=10)
    cf.fit(train_df)
    # train_recommendations, train_true_purchases = cf.predict(train_df)
    # print(cf.metrics_at_k(k=3))

    test_recommendations, test_true_purchases = cf.predict(test_df)
    print(cf.metrics_at_k(k=3))
    print(cf.explain_single_user(68570))

    # ItemCF는 435680 customer 예시
    # BPR은 68570 customer 예시

    # bpr = BPRMF(factors=150, learning_rate=0.1, regularization=0.1, iterations=100)  # best factors: 30
    # bpr.fit(train_df)
    # train_recommendations, train_true_purchases = bpr.predict(train_df)
    # print(bpr.metrics_at_k(k=3))
    #
    # test_recommendations, test_true_purchases = bpr.predict(test_df)
    # print(bpr.metrics_at_k(k=3))
    # print(bpr.explain_single_user(68570))

    # print(bpr.auc(train_df))
    # print(bpr.auc(test_df))