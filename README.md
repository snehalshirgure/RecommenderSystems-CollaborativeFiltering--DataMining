# RecommenderSystems-CollaborativeFiltering--DataMining

# Data
You will download two datasets: ml-20m.zip and ml-latest-small.zip. Once you extract the zip archives, you will find multiple data files. In this assignment, we will only use ratings.csv. However, you can combine other files to improve the performance of your recommendation system.

# Project Overview

1. Model-based CF Algorithm 

In task1, you are required to implement a Model-based CF recommendation
system by using Spark MLlib.
You are going to predict for both the small and large testing datasets mentioned above. In your code, you can set the parameters yourself to reach a better performance. You can make any improvement to your recommendation
system: speed, accuracy.
After achieving the prediction for ratings, you need to compare your result
to the correspond ground truth and compute the absolute differences. You
need to divide the absolute differences into 5 levels and count the number of
your prediction for each level.
Additionally, you need to compute the RMSE (Root Mean Squared Error)

2. User-based CF Algorithm

In this part, you are required to implement a User-based CF recommendation
system with Spark.
You are going to predict for only the small testing datasets mentioned above.
You can make any improvement to your recommendation system: speed, accuracy (e.g., Hybird approaches). It's your time to design the recommendation
system yourself, but first you need to beat the baseline.
After achieving the prediction for ratings, you need to compute the accuracy
in the same way mentioned in Model-based CF.

3. Item-based CF Algorithm

Find all the similar movies pairs in the dataset using LSH. You
need to use them to implement an item-based CF. Also measure the performance
of it as previous one does.

# Results

Model-Based CF

For small file:
>=0 and <1: 13928
>=1 and <2: 3993
>=2 and <3: 680
>=3 and <4: 125
>=4:7
RMSE: 0.948179056742
Time: 213.08100009 sec
For big file:
>=0 and <1: 3191803
>=1 and <2: 765525
>=2 and <3: 82308
>=3 and <4: 6573
>=4: 129
RMSE: 0.8385254360124645
Time: 5940. 02302031 sec

User-Based CF

For small file:
>=0 and <1: 15433
>=1 and <2: 3987
>=2 and <3: 707
>=3 and <4: 124
>=4: 5
RMSE: 0.924241691853
Time: 88.6360001564 sec

Item-Based CF

For small file with LSH:
>=0 and <1: 13640
>=1 and <2: 5106
>=2 and <3: 1247
>=3 and <4: 234
>=4: 29
RMSE: 1.05309732877
Time: 300.947999954 sec
without LSH:
>=0 and <1: 13824
>=1 and <2: 5167
>=2 and <3: 1046
>=3 and <4: 203
>=4: 16
RMSE: 1.02597760288
Time: 224.019000053 sec.

As RMSE value for Pearson correlation(without LSH) is greater than using Jaccard based LSH similarity values, it is a better algorithm.


