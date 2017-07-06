import csv
import random
import math
import numpy as np
import itertools
import json
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from sklearn import preprocessing


class Recommendation:
    def __init__(self, training_file, test_file, feature_file, test_items, training_items, recommendation_list_length, raw_item_number, factor_number):
        self.training_file = training_file
        self.test_file = test_file
        self.feature_file = feature_file
        #self.all_item_id_set = set(all_items)
        self.test_items = test_items
        self.test_item_id_set = set(self.test_items)
        # print('test items: ', self.test_item_id_set)
        self.raw_item_number = raw_item_number
        self.count(recommendation_list_length)
        self.factor_number = factor_number
        self.iteration_number = 40
        self.learning_rate = 0.01
        self.regularization = 0.15
        self.user_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in
                                   range(self.user_number)]
        self.item_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in
                                   range(self.item_number)]

        # factors for items extracted to construct the test file
        self.test_item_factor_matrix = [[random.uniform(-0.1, 0.1) for x in range(self.factor_number)] for y in
                                   range(self.test_item_number)]


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
        self.test_item_number = len(self.test_items)
        self.rating_number = len(self.rating_vector)
        summation = sum(self.rating_vector)
        self.global_mean = summation / self.rating_number
        self.mapping(recommendation_list_length)

    def mapping(self, recommendation_list_length):
        self.user_id_dictionary = dict.fromkeys(self.user_id_set, 0)
        self.item_id_dictionary = dict.fromkeys(self.item_id_set, 0)
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
        self.training_item_features = [[0] * 4096] * self.item_number
        self.test_item_features = [[0] * 4096] * self.test_item_number
        # self.features = [[0] * 4096] * self.raw_item_number
        for line in feature_file:
            s = json.loads(line)
            item_id = s['asin']
            image_feature_vector = s['image']
            try:
                if item_id in self.item_id_set:
                    self.training_item_features[self.item_id_dictionary[item_id]] = image_feature_vector
                else:
                    self.test_item_features[self.test_item_id_dictionary[item_id]] = image_feature_vector
                # self.features[self.item_id_dictionary[item_id]] = image_feature_vector
            except KeyError:
                pass
        feature_file.close()

        self.normalizer = preprocessing.Normalizer().fit(self.training_item_features)
        self.test_item_features = self.normalizer.transform(self.test_item_features)
        self.training_item_features = self.normalizer.transform(self.training_item_features)
        # print('Length of test features', len(self.test_item_features))
        # print('Length of training features', len(self.training_item_features))
        # print(self.test_item_features)

    def train(self):
        self.random_array = list(range(self.rating_number))
        random.shuffle(self.random_array)

        for i in range(self.iteration_number):
            for j in range(self.rating_number):
                index = self.random_array[j]
                user_index = self.user_vector[index][1]
                item_index = self.item_vector[index][1]
                rating = self.rating_vector[index]
                dot_product = 0.0
                for k in range(self.factor_number):
                    dot_product += self.user_factor_matrix[user_index][k] * self.item_factor_matrix[item_index][k]
                predicted_rating = dot_product + self.global_mean
                if predicted_rating > 5:
                    predicted_rating = 5
                if predicted_rating < 1:
                    predicted_rating = 1
                prediction_error = rating - predicted_rating
                for k in range(self.factor_number):
                    self.user_factor_matrix[user_index][k] += self.learning_rate * (
                        prediction_error * self.item_factor_matrix[item_index][k]
                        - self.regularization * self.user_factor_matrix[user_index][k])
                    self.item_factor_matrix[item_index][k] += self.learning_rate * (
                        prediction_error * self.user_factor_matrix[user_index][k]
                        - self.regularization * self.item_factor_matrix[item_index][k])

        self.linear()

    def linear(self):
        # print('running linear regression...')
        x = []
        y = []
        # print(self.training_item_features[0])
        # print(self.item_factor_matrix[0])
        for item_id in self.item_id_set:
            x.append(self.training_item_features[self.item_id_dictionary[item_id]])
            y.append(self.item_factor_matrix[self.item_id_dictionary[item_id]])
        # print(x)
        # print(y)
        # self.clf = linear_model.LinearRegression(fit_intercept=False)
        self.clf = linear_model.Ridge(alpha=0.5)
        self.clf.fit(x, y)
        # print(self.clf.coef_[0])
        for item_id in self.test_item_id_set:
            dim = []
            item_index = self.test_item_id_dictionary[item_id]
            for coef in self.clf.coef_:
                dim.append(np.dot(self.test_item_features[item_index], coef))
            self.test_item_factor_matrix[item_index] = dim

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
        for item in user_consumption_items:
            cnn_features.append(self.features[self.item_id_dictionary[item]])
        return cnn_features

    # generate recommendation for one user
    def generate_recommendation(self, user_id, recommendation_list_length, prefix):
        recommendation_one_user = []
        try:
            user = self.user_id_dictionary[user_id]
            for i in range(self.test_item_number):
                dot_product = 0.0
                for j in range(self.factor_number):
                    dot_product += self.user_factor_matrix[user][j] * self.test_item_factor_matrix[i][j]
                prediction = dot_product + self.global_mean
                '''if prediction > 5:
                    prediction = 5
                if prediction < 1:
                    prediction = 1'''
                prediction_struct = list([user_id, self.test_item_id_reverse_dictionary[i], prediction])
                recommendation_one_user.append(prediction_struct)
        except KeyError:
            for i in range(self.test_item_number):
                prediction = random.uniform(0, 5)
                prediction_struct = list(
                    [user_id, self.test_item_id_reverse_dictionary[i], prediction])
                recommendation_one_user.append(prediction_struct)

        recommendation_one_user = sorted(recommendation_one_user, key=lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
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
        print('CNN Ridge:\t'+str(number_in_recommendation_list) + '\t' + str(total_consumed_item_number) + '\t' +
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
    return test_items, training_items, list(set(target_users)), len(items)

def main(prefix, list_length, factor_number):
    feature_file_name = 'All_5_image_features\image_features_' + prefix + '.json'
    delimiter = ','
    training_file_name = prefix + '_training.csv'
    test_file_name = prefix + '_test.csv'
    raw_data_file_name = prefix + '.csv'

    test_items, training_items, target_users, item_number = split(raw_data_file_name, training_file_name,
                                                                  test_file_name)
    user_number = len(target_users)
    # print('Number of target users: ' + str(len(target_users)))
    # print(target_users)

    r = Recommendation(training_file_name, test_file_name, feature_file_name, test_items, training_items, list_length,
                       item_number, factor_number)

    r.recommendation_list = []
    for user in target_users:
        r.generate_recommendation_random(user, list_length)
    r.hit_ratio_random(user_number)

    n = 0
    r.recommendation_list = []
    r.train()
    for user in target_users:
        r.generate_recommendation(user, list_length, prefix)
        n += 1
    r.hit_ratio(user_number)

category = ['Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
            'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
            'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
            'Kindle_Store_5', 'Movies_and_TV_5', 'Musical_Instruments_5', 'Office_Products_5', 'Pet_Supplies_5',
            'Patio_Lawn_and_Garden_5', 'Sports_and_Outdoors_5',
            'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']
category = ['Musical_Instruments_5', 'Automotive_5', 'Digital_Music_5']
#prefix = 'Patio_Lawn_and_Garden_5'
#for list_length in [100,90,80,70,60,50,40,30,20,10]:
for prefix in category:
    '''l = []
    for i in range(10, 110, 10):
        l.append(i)
    for i in range(150, 1050, 50):
        l.append(i)'''
    print(prefix)
    for list_length in [100, 80, 60, 40, 20]:
        print(list_length)
        #for f in l:
        for f in range(10, 210, 10):
            print('Factor number\t', f)
            for i in range(5):
                # print(prefix.replace('_5', ''))
                main(prefix, list_length, f)
            print()
        print('\n')




















