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


def recommendation(algo, trainset, testset):
    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)

    # # Predictions on training set
    # train_predictions = algo.test(trainset)
    # train_rmse = accuracy.rmse(train_predictions)
    # train_mae = accuracy.mae(train_predictions)

    # Predictions on testing set
    test_predictions = algo.test(testset)
    test_rmse = accuracy.rmse(test_predictions)
    test_mae = accuracy.mae(test_predictions)

    return test_rmse, test_mae, test_predictions


file_path_train = 'training_data.csv'
file_path_test = 'testing_data.csv'
traindf = pd.read_csv(file_path_train)
testdf = pd.read_csv(file_path_test)
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)


print("1")
BaselineOnly()

algo = BaselineOnly()
test_base_rmse, test_base_mae, test_base_pred = recommendation(algo, trainset, testset)

print("2")
# basic collaborative filtering algorithm taking into account a baseline rating.
sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBaseline(sim_options=sim_options)
test_knn_rmse, test_knn_mae, test_knn_pred = recommendation(algo, trainset, testset)

print("3")
# SlopeOne
algo = SlopeOne()
test_slopeone_rmse, test_slopeone_mae, test_slopeone_pred = recommendation(algo, trainset, testset)

print("4")
# SVD
algo = SVD()
test_svd_rmse, test_svd_mae, test_svd_pred = recommendation(algo, trainset, testset)

print("5")
# SVDpp
algo = SVDpp()
test_svdpp_rmse, test_svdpp_mae, test_svdpp_pred = recommendation(algo, trainset, testset)

print("6")
test_pred_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating',
             'baseline_rating'])
test_svd_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_svdpp_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_knnb_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_slope_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
test_bonly_df = pd.DataFrame(
    columns=['uid', 'iid', 'og_rating', 'est_rating'])
num_test = len(test_base_pred)
for i in range(num_test):
    svd = test_svd_pred[i]
    slopeone = test_slopeone_pred[i]
    knn = test_knn_pred[i]
    svdpp = test_svdpp_pred[i]
    baseline = test_base_pred[i]
    df = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est, knn.est, svdpp.est, slopeone.est, baseline.est]],
                      columns=['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating',
                               'baseline_rating'])
    df_svd = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est]],
                          columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_svdpp = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svdpp.est]],
                            columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_knnb = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, knn.est]],
                           columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_slope = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, slopeone.est]],
                            columns=['uid', 'iid', 'og_rating', 'est_rating'])
    df_bonly = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, baseline.est]],
                            columns=['uid', 'iid', 'og_rating', 'est_rating'])
    test_pred_df = pd.concat([df, test_pred_df], ignore_index=True)
    test_svd_df = pd.concat([df_svd, test_svd_df], ignore_index=True)
    test_svdpp_df = pd.concat([df_svdpp, test_svdpp_df], ignore_index=True)
    test_slope_df = pd.concat([df_slope, test_slope_df], ignore_index=True)
    test_knnb_df = pd.concat([df_knnb, test_knnb_df], ignore_index=True)
    test_bonly_df = pd.concat([df_bonly, test_bonly_df], ignore_index=True)

print("7")
test_pred_df.to_csv('test_prediction_HP.csv')
test_svd_df.to_csv('test_predictions_svd.csv')
test_svdpp_df.to_csv('test_predictions_svdpp.csv')
test_knnb_df.to_csv('test_predictions_knnb.csv')
test_slope_df.to_csv('test_predictions_slope.csv')
test_bonly_df.to_csv('test_predictions_bonly.csv')
