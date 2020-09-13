import math
from collections import defaultdict
import csv
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd


def get_top_n(predictions, algo_weights, n):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    top_n_ndcg = defaultdict(list)
    for i in range(len(predictions)):
        row = predictions.iloc[i, :]
        final_est = algo_weights['svd']*float(row['svd_rating']) + algo_weights['knn']*float(row['knn_rating']) + \
                    algo_weights['svdpp']*float(row['svdpp_rating']) + algo_weights['slope']*float(row['slopeone_rating']) + \
                    algo_weights['baseline']*float(row['baseline_rating'])
        top_n[row[0]].append((row[1], final_est))
        top_n_ndcg[row[0]].append((row[1], row[2], final_est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    for uid, user_ratings in top_n_ndcg.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n_ndcg[uid] = user_ratings[:n]

    return top_n, top_n_ndcg


def precision_recall_at_k(predictions, algo_weights, k, threshold):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for i in range(len(predictions)):
        row = predictions.iloc[i, :]
        final_est = algo_weights['svd']*float(row['svd_rating']) + algo_weights['knn']*float(row['knn_rating']) + \
                    algo_weights['svdpp']*float(row['svdpp_rating']) + algo_weights['slope']*float(row['slopeone_rating']) + \
                    algo_weights['baseline']*float(row['baseline_rating'])
        user_est_true[row[0]].append((final_est, row[2]))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k/n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k/n_rel if n_rel != 0 else 1

    return precisions, recalls


def dcg_at_k(scores):
    return scores[0] + sum(sc/math.log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


def ndcg_at_k(predicted_scores, actual_scores):
    idcg = dcg_at_k(sorted(actual_scores, reverse=True))
    return (dcg_at_k(predicted_scores)/idcg) if idcg > 0.0 else 0.0


predictions = pd.read_csv("test_prediction_HP.csv", usecols=range(1, 9))
algo_weights = dict()
algo_weights['svd'] = 0
algo_weights['knn'] = 0
algo_weights['svdpp'] = 1
algo_weights['slope'] = 0
algo_weights['baseline'] = 0
n = 5
threshold = 3.75
top_n, top_n_ndcg = get_top_n(predictions, algo_weights, n)
with open('top5_svdpp.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    for key, value in top_n.items():
        writer.writerow([key, value])

ndcg_scores = dict()
for uid, user_ratings in top_n_ndcg.items():
    true = []
    est = []
    for _, tru_r, est_r in user_ratings:
        true.append(tru_r)
        est.append(est_r)
    ndcg = ndcg_at_k(est, true)
    ndcg_scores[uid] = ndcg

# Print the recommended items for each user
# for uid, user_ratings in top_n.items():
#     print(uid, [iid for (iid, _) in user_ratings])

precisions, recalls = precision_recall_at_k(predictions, algo_weights, n, threshold)
precision = sum(prec for prec in precisions.values())/len(precisions)
recall = sum(rec for rec in recalls.values())/len(recalls)
fmeasure = (2*precision*recall)/(precision + recall)
ndcg_score = sum(ndcg for ndcg in ndcg_scores.values())/len(ndcg_scores)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Measure", fmeasure)
print("NDCG Score: ", ndcg_score)
