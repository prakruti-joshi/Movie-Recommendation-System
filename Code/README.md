### Description:

#### 1. cold_start_analysis:
Analyses the performance of different approaches in case of a new user or a user with less number of interaction with the system, namely the cold start problem. Computed the rmse and mae for those customers who have rated less than 18 books and also who have rated more than 1000 movies. <br>
For less interactions, content based and item-item based collaborative filtering approaches work better. As the number of interactions per customer increases, SVD and collaborative approaches work better.

#### 2. combined_model:
Combination of different surprise model results by applying weighted linear combination to generate final rating.

#### 3. content_based_recommendation:
Genreating user and movie vectors based on genre and predicting the ratings for movies in test data.

#### 4. evaluating_recs:
Code for Precision, Recall, F-1 score and NDCG.

#### 5. generating_predictions:
Generating rating predictions for test data using surprise library.

#### 6. hybrid_model:
Code for the hybrid model based on combining recommendations from different models such as content based, CF, SVD to improve accuracy and quality of recommendations.

#### 7. knn_analysis:
Analysis of KNN algorithms by changing different parameters like:
* number of neighbors
* similarity metrices
* user v/s item based CF

#### 8. model_hyperparameter_tuning:
Fine-tuned surprise models by experimenting with different hyperparameters for training and model. Compared models based on RMSE and MAE.

#### 9. movie_era_based_recs:
Content based approach to include the time period in which the movie was launced in the user vector. This method personalizes the users recommendations to include this feature.

#### 10. movie_similarity_based_recs:
Content based approach to include the user's genre preference and recommend movies similar to user's highly rated movies.

#### 11. movie_year_analysis:
Experiments with the year of the movie release. Analysed the distribution of data and determine the appropriate era intervals to classify movies. Used the content based approach to form a user vector based on the era preference.

#### 12. popularity_model:
Model  which uses the popularity attribute as well as the average rating and voter count in the TMDB data to generate popular movies genre wise. The genres are determined using the IMDB data.

#### 13. preprocessing:
Code for spliting the data into training and testing set for each user such that 80% ratings are in training and 20% are for testing.

#### 14. surprise_model_predictions:
Code for generating ratings for test data using surprise models such as KNN (CF), SVD, Baseline approach, Slopeone etc. 

#### 15. surprise_model_recs:
Comparison between the surprise models based on test data ratings (RMSE and MAE) and quality of recommendations (precision, recall, ndcg, f-measure).

#### 16. test_ndcg:
Code to test implementation of [NDCG metric](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) for evaluting recommendations. 



