### Description:

#### 1. cold_start_analysis:
Analyses the performance of different approaches in case of a new user or a user with less number of interaction with the system, namely the cold start problem. Computed the rmse and mae for those customers who have rated less than 18 books and also who have rated more than 1000 movies. For less interactions, content based and item-item based collaborative filtering approaches work better. As the number of interactions per customer increases, SVD and collaborative approaches work better.

#### 2. combined_model:
Combination of different surprise model results by applying weighted linear combination to generate final rating.

#### 3. content_based_recommendation:
Genreating user and movie vectors based on genre and predicting the ratings for movies in test data

#### 4. generating_predictions:
Generating rating predictions for test data using surprise library

#### 5. knn_analysis:
Analysis of KNN algorithms by changing different parameters like:
* number of neighbors
* similarity metrices
* user v/s item based CF

#### 6. 
