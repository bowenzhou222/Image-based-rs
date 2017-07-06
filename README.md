# Image-based-rs
Many thanks to the ssim/cwssim code by https://github.com/jterrace/pyssim
Fisrtly use crawl_image_multi.py to download images of products from Amazon<br />
For structual similarity, use image_similarity.py to calculate pairwise similarity among all images, and then use image_based_recommendation to recommend visually similar products<br />
For CNN-feature-based recommendation, use cnn_feature_based_test_item_only_recommendation.py for KNN<br />
cnn_matrix_sgd_combined_recommendation.py for a mixed Matrix Factorization with CNN-feature-mapping recommendation<br />
cnn_ridge_factor_number_test.py uses ridge regression to map CNN features to the Matrix Factorization factors<br />

Datasets (images, reviews) required by this RS can be downloaded from http://jmcauley.ucsd.edu/data/amazon/
