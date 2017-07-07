# Image-based-rs
Many thanks to the ssim/cwssim code by https://github.com/jterrace/pyssim. Please install it before using this RS<br /><br />
Fisrtly use crawl_image_multi.py to download images of products from Amazon. It would take about 30min, depending on # of CPU cores and Internet speed. The images will be in subset/subset_images<br /><br />
For structual similarity, use image_similarity.py to calculate pairwise similarity among all images, and then use image_based_recommendation to recommend visually similar products<br /><br />
For CNN-feature-based recommendation, use cnn_feature_based_test_item_only_recommendation.py for KNN<br /><br />
cnn_matrix_sgd_combined_recommendation.py for a mixed Matrix Factorization with CNN-feature-mapping recommendation<br /><br />
cnn_ridge_factor_number_test.py uses ridge regression to map CNN features to the Matrix Factorization factors<br /><br />

Datasets (images, reviews) required by this RS can be downloaded from http://jmcauley.ucsd.edu/data/amazon/
