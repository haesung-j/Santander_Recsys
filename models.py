import warnings
import pandas as pd
import numpy as np
import multiprocessing
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from typing import Tuple

from utils import *
from metrics import *

class ItemBasedCF:
    def __init__(self, multiprocess=True, n_jobs=-1):
        # self.item_neighbors = item_neighbors
        self.multiprocess = multiprocess
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y=None):
        print("Fit model ...")
        return self._fit(X)

    def predict(self, X: pd.DataFrame) -> Tuple[dict, dict]:
        print("Generate Recommendations for all users in test set...")
        # test_df 와 X(train_df)에 모두 존재하는 Customer_id와 product_id 추출
        intersection_product_ids = set(self.prediction_result_df.columns).intersection(set(X['product_id'].unique()))
        intersection_customer_ids = set(self.prediction_result_df.index).intersection(set(X['Customer_id'].unique()))
        intersection_product_ids = sorted(list(intersection_product_ids))
        intersection_customer_ids = sorted(list(intersection_customer_ids))

        print(f"train_df와 test_df에 모두 존재하는 product_id 개수: {len(intersection_product_ids):,}개")
        print(f"train_df와 test_df에 모두 존재하는 Customer_id와 개수: {len(intersection_customer_ids):,}명")

        # train_df와 X(test_df)에 공통으로 존재하는 Customer에 대해서 추천리스트 생성
        grouped = X.groupby(by='Customer_id')
        recommendations = OrderedDict()
        true_purchases = OrderedDict()
        for customer_id, group in tqdm(grouped):
            if customer_id in intersection_customer_ids:
                already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] == 1].tolist()
                preds = self.prediction_result_df.loc[customer_id].sort_values(ascending=False).index
                rec_list = preds[~preds.isin(already_have)].values
                true_list = group['product_id'].values
                recommendations[customer_id] = list(rec_list)
                true_purchases[customer_id] = list(true_list)

        self.recommendations = recommendations
        self.true_purchase = true_purchases

        return recommendations, true_purchases

    def predict_single_user(self, customer_id):
        train_user_idx = set(self.sparse_matrix.index)
        # true_list = group['product_id'].values
        if customer_id not in train_user_idx:
            print(
                f"There is no customer_id: {customer_id} in train set. We cannot make recommendation for this customer.")
            rec_list = [999]
            scores = [0]
        else:
            already_have = self.sparse_matrix.columns[self.sparse_matrix.loc[customer_id] == 1].tolist()
            already_have_names = [prod_id_inv[p] for p in already_have]
            print(f"{customer_id}가 이미 보유한 상품: {already_have_names}")

            sim_df = self.product_sim_df.loc[already_have]
            sim_df.index = [prod_id_inv[i] for i in sim_df.index]
            sim_df.columns = [prod_id_inv[i] for i in sim_df.columns]
            print(f"보유한 상품과 다른 상품과의 유사도는 다음과 같습니다.\n{sim_df}")

            preds = self.prediction_result_df.loc[customer_id].sort_values(ascending=False)
            preds = preds[~preds.index.isin(already_have)]
            scores = preds.values
            rec_list = preds.index.values
            print(f"각 상품별 user의 구매 가능 점수는 다음과 같습니다.\n{scores}")
            print(f"따라서, 다음의 상품을 추천합니다.\n{[prod_id_inv[r] for r in rec_list]}")
        return rec_list, scores

    def mrr(self, k=3):
        score = mrr(self.true_purchase, self.recommendations, k)
        return score

    def mapk(self, k=3):
        score = mapk(self.true_purchase, self.recommendations, k)
        return score

    def mark(self, k=3):
        score = mark(self.true_purchase, self.recommendations, k)
        return score

    def hrk(self, k=3):
        score = hrk(self.true_purchase, self.recommendations, k)
        return score

    @staticmethod
    def similarity_df(x: pd.DataFrame, y=None) -> pd.DataFrame:
        if y is None:
            sim_matrix = cosine_similarity(x.values, x.values)
            sim_df = pd.DataFrame(data=sim_matrix, columns=x.index, index=x.index)
        else:
            sim_matrix = cosine_similarity(x.values, y.values)
            sim_df = pd.DataFrame(data=sim_matrix, columns=x.index, index=y.index)

        return sim_df

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        """
        1. X(pd.DataFrame)로부터 item-user sparse_matrix를 생성
        2. sparse_matrix를 기반으로 item-item similarity matrix DataFrame 생성
        3. 각 user의 구매기록을 바탕으로 item similarty간 가중합을 구한 뒤, user-item feedback 예측

        :param X: pd.DataFrame['Customer_id': int, 'product_id': int, 'purchase': int]
        :param y: None
        :return self
        """
        # 1. X(pd.DataFrame)로부터 item-user sparse_matrix를 생성
        self.sparse_matrix = pd.pivot_table(X,
                                       index='Customer_id',
                                       columns='product_id',
                                       values='purchase'
                                       ).fillna(0)

        # 2. sparse_matrix를 기반으로 item-item similarity matrix DataFrame 생성
        self.product_sim_df = self.similarity_df(self.sparse_matrix.T)  # 아이템 유사도 행렬

        # 3. 각 user의 구매기록을 바탕으로 item similarty간 가중합을 구한 뒤, user-item feedback 예측
        customer_grp = X.groupby('Customer_id')   # Customer별로 groupby

        if self.multiprocess:
            self.prediction_result_df = self._applyparallel(customer_grp, self._find)
            self.prediction_result_df = self.prediction_result_df.set_index('Customer_id')

        else:
            self.prediction_result_df = pd.DataFrame(index=customer_grp.indices.keys(), columns=self.product_sim_df.index)
            for userId, group in tqdm(customer_grp):
                product_similarities = self.product_sim_df.loc[group['product_id']]   # 유저가 구매한 상품에 대한 유사도 행렬(차원: 구매 상품 x 전체 상품)
                user_purchase = group['purchase']  # user가 구매한 상품(1) (차원: 구매한 상품 x 1)
                sim_sum = product_similarities.sum(axis=0)    # 차원: 전체 상품 x 1
                # sim_sum.loc[sim_sum == 0] = 1

                # userId 전체 purchase predictions -> 유저가 구매한 상품을 기반으로 모든 상품에 대해 유사도 가중합
                pred_ratings = np.matmul(product_similarities.T.to_numpy(), user_purchase) / (sim_sum + 1)    # 유저가 평가한 영화 vs 다른 영화들과의 유사도 %*% 유저의 평가
                self.prediction_result_df.loc[userId] = pred_ratings

    def _applyparallel(self, dfgrouped, func):
        if self.n_jobs == -1:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = self.n_jobs
        with Parallel(n_jobs=cpu_count) as parall:
            res_list = parall(delayed(func)(group) for name, group in tqdm(dfgrouped))
        return pd.concat(res_list)

    def _find(self, group):
        product_similarities = self.product_sim_df.loc[
            group['product_id']]  # customer가 구매한 상품에 대한 유사도 행렬(차원: 구매 상품 x 전체 상품)
        user_purchase = group['purchase']  # customer가 구매한 상품(1) (차원: 구매한 상품 x 1)
        sim_sum = product_similarities.sum(axis=0)  # 차원: 전체 상품 x 1
        # sim_sum.loc[sim_sum == 0] = 1

        # userId 전체 purchase predictions -> customer가 구매한 상품을 기반으로 모든 상품에 대해 유사도 가중합
        pred_ratings = np.matmul(product_similarities.T.to_numpy(), user_purchase) / (
                sim_sum + 1)  # 유저가 구매한 상품 vs 다른 상품들과의 유사도 %*% 유저의 구매여부
        pred_ratings = pd.DataFrame(pred_ratings).T
        pred_ratings['Customer_id'] = group['Customer_id'].values[0]
        return pred_ratings



if __name__ == '__main__':
    DATA_PATH = './data'
    product_table = load_product_table(DATA_PATH)
    prod_id, prod_id_inv = get_product_dictionary(product_table)
    train_df = get_implicit_binary(product_table)  # 4월에 고객별 보유중인 상품
    test_df = get_test_set(product_table)  # 5월에 새로 구매한 (고객,상품)

    print(train_df.shape)

    cf = ItemBasedCF(multiprocess=True, n_jobs=-1)
    cf.fit(train_df.sample(20000, random_state=0))
    recommendations, true_purchases = cf.predict(test_df)
    print(recommendations)
    print(true_purchases)
    print(cf.mapk(), cf.mark(), cf.hrk(), cf.mrr())
    print(cf.predict_single_user(435680))