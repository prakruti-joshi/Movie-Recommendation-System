import math
from collections import defaultdict

from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise.model_selection import train_test_split

import pandas as pd
import numpy as np


def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset


# Modified get_top_n function   -----------------------------------
# actual_ratings: list of actual ratings for all iids for each user
def get_top_n(predictions, n):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    org_ratings = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        org_ratings[uid].append((iid, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n, org_ratings
# -------------------------------------------------------------------


def dcg_at_k(scores):
    return scores[0] + sum(sc/math.log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


# Modified to include only one parameter-------------------------------
def ndcg_at_k(scores):
    idcg = dcg_at_k(sorted(scores, reverse=True))
    return (dcg_at_k(scores)/idcg) if idcg > 0.0 else 0.0
# ---------------------------------------------------------------------


file_path_train = 'training_data.csv'
file_path_test = 'testing_data.csv'
traindf = pd.read_csv(file_path_train)
testdf = pd.read_csv(file_path_test)
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)

print("Starting algo")
algo = SVDpp()
algo.fit(trainset)
test_predictions = algo.test(testset)
test_rmse = accuracy.rmse(test_predictions)
test_mae = accuracy.mae(test_predictions)
print("Ended algo")

top_n, org_ratings = get_top_n(test_predictions, 5)   # --------------- Modified this line

ndcg_scores = dict()

# Modified----------------------
for uid, user_ratings in top_n.items():
    scores = []
    for iid, est_r in user_ratings:
        iid_found = False
        org_user_ratings = org_ratings[uid]
        for i, r in org_user_ratings:
            if iid == i:
                scores.append(r)
                iid_found = True
                break
        if not iid_found:
            scores.append(0)
    ndcg_scores[uid] = ndcg_at_k(scores)
# --------------------------------

ndcg_score = sum(ndcg for ndcg in ndcg_scores.values())/len(ndcg_scores)
print(ndcg_score)
