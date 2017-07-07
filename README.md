# Image-based-rs
Many thanks to the ssim/cwssim code by https://github.com/jterrace/pyssim. Please install it before using this RS<br /><br />
Fisrtly use crawl_image_multi.py to download images of products from Amazon. It would take about 30min, depending on # of CPU cores and Internet speed. The images will be in subset/subset_images<br /><br />
For structual similarity, use image_similarity.py to calculate pairwise similarity among all images, and then use image_based_recommendation to recommend visually similar products.<br />
Since it would take some time to calculate pairwise similarity, the results are provided in subset/image_similarity/<br /><br />
The output of the following format:<br />
Dataset name<br />
Random: #-items-recommended&nbsp;&nbsp;&nbsp;&nbsp;#-items-consumed&nbsp;&nbsp;&nbsp;&nbsp;hit-ratio<br />
ssim: #-items-recommended&nbsp;&nbsp;&nbsp;&nbsp;#-items-consumed&nbsp;&nbsp;&nbsp;&nbsp;hit-ratio<br />
cwssim: #-items-recommended&nbsp;&nbsp;&nbsp;&nbsp;#-items-consumed&nbsp;&nbsp;&nbsp;&nbsp;hit-ratio<br />




For CNN-feature-based recommendation, use cnn_feature_based_test_item_only_recommendation.py for KNN<br /><br />
The output of CNN-feature-based recommendation has the following format:<br />
length of recommendation list<br />
Dataset name<br />
Random: #-items-recommended&nbsp;&nbsp;&nbsp;&nbsp;#-items-consumed&nbsp;&nbsp;&nbsp;&nbsp;hit-ratio<br />
CNN: #-items-recommended&nbsp;&nbsp;&nbsp;&nbsp;#-items-consumed&nbsp;&nbsp;&nbsp;&nbsp;hit-ratio<br />






cnn_matrix_sgd_combined_recommendation.py for a mixed Matrix Factorization with CNN-feature-mapping recommendation<br /><br />


cnn_ridge_factor_number_test.py uses ridge regression to map CNN features to the Matrix Factorization factors<br /><br />

Datasets (images, reviews) required by this RS can be downloaded from http://jmcauley.ucsd.edu/data/amazon/
