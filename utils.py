import os
import warnings
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict
from pathlib import Path

warnings.filterwarnings('ignore')


def load_product_table(data_path: Path) -> pd.DataFrame:
    # customer_path = os.path.join(data_path, 'bio_table.csv')
    # status_path = os.path.join(data_path, 'status_table.csv')
    product_path = os.path.join(data_path, 'product_table.csv')

    product_tab = pd.read_csv('./data/product_table.csv')

    # 상품 보유 정보가 결측치인 경우 미보유(0)로 대체
    for col in product_tab.columns:
        if product_tab[col].isna().sum() > 0:
            product_tab[col] = product_tab[col].fillna(0).astype(int)

    return product_tab


def get_product_dictionary(product_table: pd.DataFrame) -> Tuple[dict, dict]:
    prod_id = {prod: i for i, prod in enumerate(product_table.columns[2:])}
    prod_id_inv = {v: k for k, v in prod_id.items()}
    return prod_id, prod_id_inv


def get_implicit_binary(product_table: pd.DataFrame) -> pd.DataFrame:
    # 데이터 상 가장 마지막 달(test)과 이전 달 추출
    date = pd.to_datetime(product_table['Fetch_date']).unique()
    last_month = str(date.max().date())
    prev_month = str((date.max() - pd.DateOffset(months=1)).date())
    prod_id, prod_id_inv = get_product_dictionary(product_table)

    # 마지막 달(test month)을 제외한 나머지만 추출
    prod_train = product_table[product_table['Fetch_date'] != last_month]

    prod_train = prod_train.sort_values(by=['Customer_id', 'Fetch_date'])
    prod_train = prod_train.groupby('Customer_id').last()
    prod_train = prod_train.drop('Fetch_date', axis=1).reset_index(drop=False)
    # prod_train = prod_train.set_index('Customer_id')

    prod_train_df = pd.melt(prod_train, id_vars='Customer_id', value_vars=prod_train.columns.tolist()[1:])
    prod_train_df['product_id'] = prod_train_df['variable'].map(prod_id)
    prod_train_df = prod_train_df.drop('variable', axis=1)
    prod_train_df = prod_train_df[['Customer_id', 'product_id', 'value']].rename(columns={'value': 'purchase'})
    prod_train_df = prod_train_df[prod_train_df['purchase'] > 0].reset_index(drop=True)
    # status_train = status_tab[status_tab['Fetch_date'] != last_month]
    # status_train = status_train.sort_values(by=['Customer_id', 'Fetch_date'])
    # status_train = status_train.groupby('Customer_id').last()
    return prod_train_df


def get_implicit_count(product_table: pd.DataFrame, train_months: int = 5) -> pd.DataFrame:
    """
    주어진 train_months 기간 동안 상품 보유한 기간을 count한 implicit feedback datatframe 생성
    :param product_table: 상품 보유 정보가 sparse한 형태로 저장되어 있는 데이터 프레임 (cols: Customer_id, Fetch_date, products)
    :param train_months: 과거 몇 달동안의 feedback을 사용할 것인지 결정하는 요소로, 각 상품 count의 max값이 됨
    :return: prod_train_df: 고객별 각 상품별 보유 기간을 월단위로 count (cols: Customer_id, product_id, purchase)
    """
    # 데이터 상 가장 마지막 달(test)과 이전 달 추출
    date = pd.to_datetime(product_table['Fetch_date']).unique()
    last_month = str(date.max().date())
    train_months = ([str((date.max() - pd.DateOffset(months=i)).date()) for i in range(1, train_months+1)])[::-1]
    prod_id, prod_id_inv = get_product_dictionary(product_table)
    print(f"Train 기간: {train_months}, Test 기간: {last_month}")

    # 1. feedback count 시작하는 달에 변동이 일어났던 고객 인덱스를 추출한 뒤
    prod_train = product_table[product_table['Fetch_date'].isin(train_months)]
    first_month_cust_idx = set(product_table[product_table['Fetch_date'] == train_months[0]]['Customer_id'])

    # 2. 변동이 이전에 일어났던 고객을 따로 추출(마지막 변동이 train_months[0] 달에 고객 보유 현황이 됨)
    rest_customer = product_table[product_table['Fetch_date'] < train_months[0]]  # 이전 변동 기록 중에서
    rest_customer = rest_customer[~rest_customer['Customer_id'].isin(first_month_cust_idx)]  # 변동이 이전에 일어난 고객 추출
    rest_customer = rest_customer.sort_values(by=['Customer_id', 'Fetch_date'])  # 고객별 날짜 순으로 정렬한 뒤
    rest_customer = rest_customer.groupby('Customer_id').last().reset_index()  # 마지막 변동을 가져옴

    # 3. 고객 merge
    prod_train = pd.concat([prod_train, rest_customer], axis=0, ignore_index=True)

    # 4. 고객별 각 상품의 월별 보유 여부 sum -> train_months 동안의 각 상품별 보유기간
    prod_train = prod_train.drop('Fetch_date', axis=1).groupby('Customer_id', as_index=False).sum()

    # 5. sparse 형태를 stack 형태로 melt
    prod_train_df = pd.melt(prod_train, id_vars='Customer_id', value_vars=prod_train.columns.tolist()[1:])
    prod_train_df['product_id'] = prod_train_df['variable'].map(prod_id)
    prod_train_df = prod_train_df.drop('variable', axis=1)
    prod_train_df = prod_train_df[['Customer_id', 'product_id', 'value']].rename(columns={'value': 'purchase'})
    prod_train_df = prod_train_df[prod_train_df['purchase'] > 0].reset_index(drop=True)   # 상품을 보유하지 않은 관측치 제외

    return prod_train_df


def get_test_set(product_table: pd.DataFrame) -> pd.DataFrame:
    """
    마지막 달에 구매가 일어난 고객의 새로 구매한 상품만 추출
    columns: [Customer_id, product_id, purchase]
    """
    # 상품 딕셔너리
    prod_id, prod_id_inv = get_product_dictionary(product_table)

    # 1. 마지막 달에 구매기록 존재하는 고객 추출
    date = pd.to_datetime(product_table['Fetch_date']).unique()
    last_month = str(date.max().date())
    product_last_month = product_table[product_table['Fetch_date'] == last_month]

    # 2. 마지막 달 직전 보유 상품 추출
    product_before_last_month = product_table[product_table['Fetch_date'] != last_month]
    product_before_last_month = product_before_last_month.sort_values(by=['Customer_id', 'Fetch_date'])
    product_before_last_month = product_before_last_month.groupby('Customer_id').last().reset_index()

    # 3. 마지막 달 구매기록이 존재하는 고객의 직전 보유 상품 추출(마지막 달 구매기록이 없다면 상품 그대로 유지이기 때문)
    variant_customers = product_before_last_month[
        product_before_last_month['Customer_id'].isin(set(product_last_month['Customer_id']))]

    # 4. 마지막 달 직전 구매기록을 마지막달 구매기록에 merge
    variant_customers['Fetch_date'] = last_month
    variant_customers.columns = ['Customer_id', 'Fetch_date'] + [col + '_prev' for col in prod_id.keys()]
    product_last_month = product_last_month.merge(variant_customers, how='left')

    # 5. 기존 고객들과 마지막 달에만 기록되어 이전 기록이 없는 고객 구분(new_customer는 CF를 통해 예측할 수 없음)
    old_customers = product_last_month[product_last_month['Savings_account_prev'].notnull()]
    new_customers = product_last_month[product_last_month['Savings_account_prev'].isnull()]

    # 6. 이전 기록이 없는 고객 movielens 데이터 형태 변환
    new_customers = pd.melt(frame=new_customers,
                            id_vars='Customer_id',
                            value_vars=prod_id.keys()
                            )
    new_customers = new_customers[new_customers['value'] == 1]
    new_customers.columns = ['Customer_id', 'product_id', 'purchase']
    new_customers['product_id'] = new_customers['product_id'].map(prod_id)

    # 7. 이전 기록이 있는 고객 movielens 데이터 형태 변환
    Xs = []
    ys = []
    for product, idx in prod_id.items():
        prev_col = product + '_prev'
        X = old_customers[(old_customers[product] == 1) & (old_customers[prev_col] == 0)]
        y = np.zeros(X.shape[0], dtype=np.int8) + idx

        Xs.append(X)
        ys.append(y)

    data = pd.concat(Xs)
    y = np.hstack(ys)
    data['y'] = y

    new_purchase_df = data[['Customer_id', 'y']]
    new_purchase_df.columns = ['Customer_id', 'product_id']
    new_purchase_df['purchase'] = 1

    # 8. 데이터 병합
    new_purchase_df = pd.concat([new_purchase_df, new_customers], axis=0, ignore_index=True)

    return new_purchase_df


if __name__ == '__main__':
    DATA_PATH = './data'
    product_table = load_product_table(DATA_PATH)
    train_df_binary = get_implicit_binary(product_table)    # 4월에 고객별 보유중인 상품
    train_df_count = get_implicit_count(product_table, train_months=5)  # 2015/12 - 2015/4월에 고객별 보유중인 상품
    test_df = get_test_set(product_table)            # 5월에 새로 구매한 (고객,상품)
    print(train_df_binary.shape)
    print(train_df_binary.head())
    print(train_df_count.shape)
    print(train_df_count.head())
    print(test_df.shape)
    print(test_df.head())