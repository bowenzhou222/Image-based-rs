import csv
import random
import math
import numpy as np
import itertools
import json
from sklearn.neighbors import NearestNeighbors


class Recommendation:
    def __init__(self, training_file, test_file, feature_file, test_items, all_items, recommendation_list_length, raw_item_number):
        self.training_file = training_file
        self.test_file = test_file
        self.feature_file = feature_file
        self.test_items = test_items
        self.test_item_id_set = set(self.test_items)
        self.test_item_number = len(self.test_item_id_set)
        self.all_item_id_set = set(all_items)
        self.raw_item_number = raw_item_number
        self.count(recommendation_list_length)
        self.factor_number = 1
        self.iteration_number = 20
        self.learning_rate = 0.01
        self.regularization = 0.15
        # self.user_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.user_number)]
        # self.item_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.item_number)]
        # self.user_factor_matrix_net = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.user_number)]
        # self.item_factor_matrix_net = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in range(self.item_number)]

    def count(self, recommendation_list_length):
        self.user_vector = list()
        self.item_vector = list()
        self.item_category_vector = list()
        self.rating_vector = list()
        self.rating_vector_net = list()
        self.rating_helpful_number = list()
        self.rating_time = list()
        with open(self.training_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter=','))
            for line in csv_lines:
                if not line:
                    continue
                self.rating_time.append(int(line[3]))
        min_time = min(self.rating_time)
        with open(self.training_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter=','))
            for line in csv_lines:
                if not line:
                    continue
                # print (line)
                self.user_vector.append(str(line[0]))
                self.item_vector.append(str(line[1]))
                self.rating_vector.append(float(line[2]))
                # self.rating_vector.append(float(line[2])*(1+alpha*math.log(1+5*float(line[4]))))
                self.rating_helpful_number.append(int(line[4]))
        self.rating_vector = list(map(float, self.rating_vector))
        self.rating_vector_net = list(map(float, self.rating_vector_net))
        # {'B0AFNKIQ':'Movie', 'B02ANSDCOI':'Books'......}
        self.item_category_dictionary = dict(zip(self.item_vector, self.item_category_vector))
        self.user_id_set = set(self.user_vector)
        self.item_id_set = set(self.item_vector)
        self.user_number = len(self.user_id_set)
        self.item_number = len(self.item_id_set)
        self.rating_number = len(self.rating_vector)
        summation = sum(self.rating_vector)
        self.global_mean = summation / self.rating_number
        self.mapping(recommendation_list_length)

    def mapping(self, recommendation_list_length):
        self.user_id_dictionary = dict.fromkeys(self.user_id_set, 0)
        self.item_id_dictionary = dict.fromkeys(self.all_item_id_set, 0)
        self.test_item_id_dictionary = dict.fromkeys(self.test_item_id_set, 0)
        self.user_id_reverse_dictionary = dict()
        self.item_id_reverse_dictionary = dict()
        self.test_item_id_reverse_dictionary = dict()

        num = 0
        for u_d in self.user_id_dictionary:
            self.user_id_dictionary[u_d] = num
            self.user_id_reverse_dictionary[num] = u_d
            num += 1
        num = 0
        for i_d in self.item_id_dictionary:
            self.item_id_dictionary[i_d] = num
            self.item_id_reverse_dictionary[num] = i_d
            num += 1
        num = 0
        for i_d in self.test_item_id_dictionary:
            self.test_item_id_dictionary[i_d] = num
            self.test_item_id_reverse_dictionary[num] = i_d
            num += 1

        self.user_vector = list(
            map(lambda x: [x, self.user_id_dictionary[x]], self.user_vector))  # [[user_id, index],[],[],[]...]
        self.item_vector = list(
            map(lambda x: [x, self.item_id_dictionary[x]], self.item_vector))  # [[item_id, index],[],[],[]...]

        feature_file = open(self.feature_file, 'r')
        self.features = [[0] * 4096] * self.raw_item_number
        self.test_item_features = [[0] * 4096] * (self.raw_item_number - self.item_number)
        for line in feature_file:
            s = json.loads(line)
            item_id = s['asin']
            image_feature_vector = s['image']
            try:
                self.features[self.item_id_dictionary[item_id]] = image_feature_vector
                if item_id not in self.item_id_set:
                    self.test_item_features[self.test_item_id_dictionary[item_id]] = image_feature_vector
            except KeyError:
                pass
        feature_file.close()

        self.nbrs = NearestNeighbors(n_neighbors=recommendation_list_length, algorithm='ball_tree').fit(self.test_item_features)
        # print('Length of features', len(self.features))
        '''for i in range(self.item_number):
            print(self.features[i])'''

    def item_category_of_user(self, user_id):
        # find the index of user_id in self.user_vector
        user_index = [i for i, x in enumerate(self.user_vector) if x[0] == user_id]
        user_training_item_category = list(set(list(map(lambda x: self.item_category_vector[x], user_index))))
        return user_training_item_category

    def consumption_of_user(self, user_id):
        # find the indices of user_id in self.user_vector
        user_index = [i for i, x in enumerate(self.user_vector) if x[0] == user_id]
        l = list(map(lambda x: [self.item_vector[x][0], self.rating_vector[x]], user_index))
        user_consumption = list(l)
        return user_consumption

    # store cnn features of items in user_consumption[:]
    def store_cnn_features(self, user_consumption, prefix):
        user_consumption_items = list(map(lambda x: x[0], user_consumption))
        cnn_features = []
        '''feature_file_name = 'All_5_image_features\image_features_'+prefix+'.json'
        feature_file = open(feature_file_name, 'r')
        for line in feature_file:
            s = json.loads(line)  # s = {'asin':'xxxx', 'image':[x,x,x,x....]}
            if s['asin'] in user_consumption_items:
                cnn_features.append([s['asin'], s['image']])
        feature_file.close()'''
        for item in user_consumption_items:
            cnn_features.append(self.features[self.item_id_dictionary[item]])
        return cnn_features

    # generate recommendation for one user
    def generate_recommendation(self, user, recommendation_list_length, prefix):
        recommendation_one_user = []
        user_consumption = self.consumption_of_user(user)
        user_consumption_cnn_features = self.store_cnn_features(user_consumption, prefix)
        # print('cnn features example:\t', user_consumption_cnn_features[0])
        # calculate the mean cnn features of this user by averaging cnn feature vector of consumed items of this user
        if len(user_consumption_cnn_features) == 0:
            # print('This user has not consumed any product', end=' ')
            mean_user_consumption_cnn_feature = np.asarray([0.0] * 4096)
        else:
            mean_user_consumption_cnn_feature = np.average(user_consumption_cnn_features, axis=0)
        # print(mean_user_consumption_cnn_feature)

        # find KNN of this user's mean cnn features out of all items
        # self.nbrs = NearestNeighbors(n_neighbors=recommendation_list_length, algorithm='ball_tree').fit(self.features)
        distances, indices = self.nbrs.kneighbors([mean_user_consumption_cnn_feature])
        distances = distances[0]  # distance of neighbours to the mean_user_consumption_cnn_feature
        indices = indices[0]  # index of nearest neighbours of mean_user_consumption_cnn_feature in self.features
        for i in range(recommendation_list_length):
            item = self.test_item_id_reverse_dictionary[indices[i]]
            distance = distances[i]
            recommendation_one_user.append([user, item, distance])  # asin of the nearest neighbours
        self.recommendation_list.append(recommendation_one_user)

    def generate_recommendation_random(self, user, recommendation_list_length):
        recommendation_one_user = list()
        for i in range(self.test_item_number):
            prediction = random.uniform(0, 5)
            prediction_struct = list(
                [user, self.test_item_id_reverse_dictionary[i], prediction])
            recommendation_one_user.append(prediction_struct)
        recommendation_one_user = sorted(recommendation_one_user, key=lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
        self.recommendation_list.append(recommendation_one_user)

    def hit_ratio(self, recommendation_user_number):
        csv_file = open(self.test_file, 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()

        item_set = set(list(map(lambda x: x[1], test_recommendation_list)))
        # print('Test user number: ', recommendation_user_number)
        number_in_recommendation_list = 0
        total_consumed_item_number = 0

        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]  # list for a specific user
            # list of consumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            # print(consumed_list)  # [[user, item, rating, ..],[],[]...]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    number_in_recommendation_list += 1
        print('CNN:\t'+str(number_in_recommendation_list) + '\t' + str(total_consumed_item_number) + '\t' +
              str(float(number_in_recommendation_list / total_consumed_item_number)), end='\n', flush=True)

    def hit_ratio_random(self, recommendation_user_number):
        csv_file = open(self.test_file, 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()

        item_set = set(list(map(lambda x: x[1], test_recommendation_list)))
        # print('Test user number: ', recommendation_user_number)
        number_in_recommendation_list = 0
        total_consumed_item_number = 0

        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]  # list for a specific user
            # list of consumed products of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    number_in_recommendation_list += 1
        print('Random:\t' + str(number_in_recommendation_list) + '\t' + str(total_consumed_item_number) + '\t' +
              str(float(number_in_recommendation_list / total_consumed_item_number)), end='\n', flush=True)


def split(rating_file, training_file, test_file):
    # read from the raw rating file
    fr = open(rating_file, 'r')
    items = []
    for line in fr:
        line_split = line.split(',')
        items.append(str(line_split[1]))
    fr.close()
    items = list(set(items))  # all items
    number_test_items = int(len(items) * 0.5)
    # print('Number of test items: ' + str(number_test_items))
    # randomly choose some items
    random.shuffle(items)
    test_items = items[:number_test_items]
    training_items = items[number_test_items:]
    target_users = []
    fr = open(rating_file, 'r')
    f_training = open(training_file, 'w')
    f_test = open(test_file, 'w')
    n = 0
    for line in fr:
        line_split = line.split(',')
        if str(line_split[1]) in test_items:  # write into test file
            n += 1
            target_users.append(str(line_split[0]))
            f_test.write(line)
        else:  # write into training file
            f_training.write(line)
    # print('Number of test ratings: ' + str(n))
    fr.close()
    f_training.close()
    f_test.close()
    return items, test_items, training_items, list(set(target_users)), len(items)

def main(prefix, list_length):
    feature_file_name = 'All_5_image_features\image_features_' + prefix + '.json'
    delimiter = ','
    training_file_name = prefix + '_training.csv'
    test_file_name = prefix + '_test.csv'
    raw_data_file_name = prefix + '.csv'

    all_items, test_items, training_items, target_users, item_number = split(raw_data_file_name, training_file_name,
                                                                             test_file_name)
    user_number = len(target_users)
    # print('Number of target users: ' + str(len(target_users)))
    # print(target_users)

    r = Recommendation(training_file_name, test_file_name, feature_file_name, test_items, all_items, list_length,
                       item_number)

    r.recommendation_list = []
    for user in target_users:
        r.generate_recommendation_random(user, list_length)
    r.hit_ratio_random(user_number)

    n = 0
    r.recommendation_list = []
    for user in target_users:
        r.generate_recommendation(user, list_length, prefix)
        n += 1
        # print(n)

    r.hit_ratio(user_number)


category = ['Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
            'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
            'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
            'Kindle_Store_5', 'Movies_and_TV_5', 'Musical_Instruments_5', 'Office_Products_5', 'Pet_Supplies_5',
            'Patio_Lawn_and_Garden_5', 'Sports_and_Outdoors_5',
            'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']

category = ['Musical_Instruments_5', 'Patio_Lawn_and_Garden_5', 'Automotive_5',
            'Office_Products_5', 'Digital_Music_5', 'Baby_5']

for list_length in [100,90,80,70,60,50,40,30,20,10]:
    print(list_length)
    for i in range(5):
        for prefix in category:
            print(prefix.replace('_5', ''))
            main(prefix, list_length)
        print('\n')
























