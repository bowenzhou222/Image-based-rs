import csv
import random
import math
import numpy as np
import itertools
import ssim.ssimlib as pyssim
from ssim.utils import get_gaussian_kernel


class Recommendation:
    def __init__(self, training_file, test_file, global_avg_sim, similarity_info):
        self.training_file = training_file
        self.test_file = test_file
        self.global_avg_sim = global_avg_sim
        self.similarity_info = similarity_info[:]
        self.recommendation_list = []
        self.count()

    def count(self):
        self.user_vector = list()
        self.item_vector = list()
        self.rating_vector = list()

        with open(self.training_file, newline='') as csvfile:
            csv_lines = list(csv.reader(csvfile, delimiter=','))
            for line in csv_lines:
                if not line:
                    continue
                # print (line)
                self.user_vector.append(str(line[0]))
                self.item_vector.append(str(line[1]))
                self.rating_vector.append(float(line[2]))
        self.user_id_set = set(self.user_vector)
        self.item_id_set = set(self.item_vector)
        self.user_number = len(self.user_id_set)
        self.item_number = len(self.item_id_set)
        self.rating_number = len(self.rating_vector)
        self.global_mean = sum(self.rating_vector) / self.rating_number
        self.mapping()

    def mapping(self):
        # self.user_id_dictionary = dict.fromkeys(self.user_id_set, set(range(0, self.user_number)))
        # self.item_id_dictionary = dict.fromkeys(self.item_id_set, set(range(0, self.item_number)))
        self.user_id_dictionary = dict.fromkeys(self.user_id_set, 0)
        self.item_id_dictionary = dict.fromkeys(self.item_id_set, 0)
        self.user_id_reverse_dictionary = dict()
        self.item_id_reverse_dictionary = dict()

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

        self.user_vector = list(map(lambda x: [x, self.user_id_dictionary[x]], self.user_vector))
        self.item_vector = list(map(lambda x: [x, self.item_id_dictionary[x]], self.item_vector))  # [item_id, position in dict]

    # prevent the recommendation_user_number from > total number of users in the training set
    def check_user_number(self, recommendation_user_number):
        return min(recommendation_user_number, len(self.user_id_set))

    def consumption_of_user(self, user_id):
        # find the indices of user_id in self.user_vector
        user_index = [i for i, x in enumerate(self.user_vector) if x[0] == user_id]
        l = list(map(lambda x: [self.item_vector[x][0], self.rating_vector[x]], user_index))
        user_consumption = list(l)
        return user_consumption

    # generate recommendation for one user according to image similarity
    def generate_recommendation(self, user, recommendation_list_length, r):
        recommendation_one_user = list()
        # for item_id in self.item_id_set:
        # calculate similarity between target item and each consumed item, choose the higher similarity
        similarity_dict = dict.fromkeys(self.item_id_set, 0)
        user_id = self.user_id_reverse_dictionary[user]
        user_consumption = self.consumption_of_user(user_id)
        for line in self.similarity_info:  # search through the similarity file
            for consumption, rating in user_consumption:
                if consumption == line[0]:
                    try:
                        if r:
                            if similarity_dict[line[1]] == 0:
                                similarity_dict[line[1]] = float(line[2])*rating
                            else:
                                similarity_dict[line[1]] = (float(similarity_dict[line[1]])+float(line[2])*rating)/2
                            #similarity_dict[line[1]] = max(float(similarity_dict[line[1]]), float(line[2])*rating)
                        else:
                            #similarity_dict[line[1]] = max(float(similarity_dict[line[1]]), float(line[2]))
                            if similarity_dict[line[1]] == 0:
                                similarity_dict[line[1]] = float(line[2])
                            else:
                                similarity_dict[line[1]] = (float(similarity_dict[line[1]])+float(line[2]))/2
                        # print(float(line[2]))
                    except KeyError:
                        pass
                elif consumption == line[1]:
                    try:
                        if r:
                            if similarity_dict[line[0]] == 0:
                                similarity_dict[line[0]] = float(line[2])*rating
                            else:
                                similarity_dict[line[0]] = (float(similarity_dict[line[0]])+float(line[2])*rating)/2
                            #similarity_dict[line[0]] = max(float(similarity_dict[line[0]]), float(line[2])*rating)
                        else:
                            if similarity_dict[line[0]] == 0:
                                similarity_dict[line[0]] = float(line[2])
                            else:
                                similarity_dict[line[0]] = (float(similarity_dict[line[0]])+float(line[2]))/2
                            #similarity_dict[line[0]] = max(float(similarity_dict[line[0]]), float(line[2]))
                    except KeyError:
                        pass

        '''for item_id in self.item_id_set:
            if similarity_dict[item_id] == 0:
                similarity_dict[item_id] = self.global_avg_sim'''

        for item_id in self.item_id_set:
            similarity = similarity_dict[item_id]
            prediction_struct = list(
                [self.user_id_reverse_dictionary[user], item_id, similarity])
            recommendation_one_user.append(prediction_struct)
        recommendation_one_user = sorted(recommendation_one_user, key=lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
        self.recommendation_list.append(recommendation_one_user)
        # print(self.recommendation_list)

    def generate_recommendation_random(self, user, recommendation_list_length):
        recommendation_one_user = list()
        for i in range(self.item_number):
            prediction = random.uniform(0, 5)
            prediction_struct = list(
                [self.user_id_reverse_dictionary[user], self.item_id_reverse_dictionary[i], prediction])
            recommendation_one_user.append(prediction_struct)
        recommendation_one_user = sorted(recommendation_one_user, key=lambda x: x[2], reverse=True)
        recommendation_one_user = recommendation_one_user[0:recommendation_list_length]
        self.recommendation_list.append(recommendation_one_user)

    def hit_ratio(self, mode, recommendation_user_number):
        csv_file = open(self.test_file, 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()

        item_set = set(list(map(lambda x: x[1], test_recommendation_list)))
        recommendation_user_number = min(recommendation_user_number, len(item_set))
        #print('Test user number: ', recommendation_user_number)
        number_in_recommendation_list = 0
        total_consumed_item_number = 0

        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]  # list for a specific user
            # list of consumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    number_in_recommendation_list += 1
        print(mode+':\t'+str(number_in_recommendation_list)+'\t'+str(total_consumed_item_number)+'\t'+
              str(float(number_in_recommendation_list / total_consumed_item_number)), end='\n', flush=True)

    def hit_ratio_random(self, mode, recommendation_user_number):
        csv_file = open(self.test_file, 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()

        item_set = set(list(map(lambda x: x[1], test_recommendation_list)))
        recommendation_user_number = min(recommendation_user_number, len(item_set))
        #print('Test user number: ', recommendation_user_number)
        number_in_recommendation_list = 0
        total_consumed_item_number = 0

        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]  # list for a specific user
            # list of consumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    number_in_recommendation_list += 1
        '''print(mode+': '+str(number_in_recommendation_list)+'\t'+str(total_consumed_item_number)+'\t'+
              str(float(number_in_recommendation_list / total_consumed_item_number)), end='\n', flush=True)'''
        return number_in_recommendation_list, total_consumed_item_number, \
               float(number_in_recommendation_list / total_consumed_item_number)

    def ranking(self, recommendation_user_number):
        csv_file = open(self.test_file, 'r')
        test_recommendation_list = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()
        number_in_recommendation_list = 0
        total_consumed_item_number = 0
        rank = list()
        for i in range(recommendation_user_number):
            recommendation_list = self.recommendation_list[i]  # list for a specific user
            # list of consumed product of this specific user
            consumed_list = [x for x in test_recommendation_list if x[0] == recommendation_list[0][0]]
            consumed_item_number = len(consumed_list)
            total_consumed_item_number += consumed_item_number
            for j in range(consumed_item_number):
                consumed_item_id = consumed_list[j][1]
                check_list = list(map(lambda x: consumed_item_id in x, recommendation_list))
                if True in check_list:
                    rank.append(int(check_list.index(True)))
        print(sum(rank) / len(rank))


#############################
# split dataset into test/training = ratio
def split(raw_data_file_name, training_file_name, test_file_name, delimiter):
    training_file = open(training_file_name, 'wt')
    test_file = open(test_file_name, 'wt')
    with open(raw_data_file_name, newline='') as csvfile:
        csv_lines = list(csv.reader(csvfile, delimiter=delimiter))
        line_number = len(list(csv_lines))
        random.shuffle(csv_lines)
        # csv_lines.sort(key=lambda x: float(x[3]), reverse=True)
        # random_list = list(range(line_number))
        # random.shuffle(random_list)
        ratio = 0.6
        #print('test ratio: ' + str(ratio) + ' list length: 100')
        test_line_number = ratio * line_number
        for i in range(line_number):
            # index = random_list[i]
            # line = csv_lines[index]
            line = csv_lines[i]
            if i < test_line_number:
                test_file.write(
                    line[0] + ',' + line[1] + ',' + line[2] + ',' + line[3] + ',' + line[4] + ',' + line[5] + '\n')
            else:
                training_file.write(
                    line[0] + ',' + line[1] + ',' + line[2] + ',' + line[3] + ',' + line[4] + ',' + line[5] + '\n')

    training_file.close()
    test_file.close()


def global_average_similarity(similarity_file_path):
    f = open(similarity_file_path, 'r')
    lines = list(csv.reader(f, delimiter=','))
    similarity = []
    for line in lines:
        similarity.append(float(line[2]))
    f.close()
    return np.average(similarity), lines


def recommend(prefix, ssim_mode, random_recommendation, sp, length):
    image_path = r'd:\Study\research\amazon\experiment\json data\subset\subset_images\\'
    image_similarity_file_path = r'd:\Study\research\amazon\experiment\json data\subset\image_similarity\\'
    rating_file_path = r'd:\Study\research\amazon\experiment\json data\subset\\'

    # global_avg_sim: the global average ssim/cw-ssim coefficient between all pairs of image in the category
    # similarity_info: list of similarity of the form [[item1, item2, similarity], ...]
    global_avg_sim, similarity_info = global_average_similarity(
        image_similarity_file_path + prefix + '_' + ssim_mode + '.csv')
    #print('Global average similarity: ', global_avg_sim)
    training_file_name = '5_training.csv'
    test_file_name = '5_test.csv'

    delimiter = ','
    raw = rating_file_path + prefix + '_sub_dense.csv'
    t1 = rating_file_path + prefix + '_sub_dense_' + training_file_name
    t2 = rating_file_path + prefix + '_sub_dense_' + test_file_name
    if sp:
        split(raw, t1, t2, delimiter)

    r = Recommendation(t1, t2, global_avg_sim, similarity_info)
    recommendation_user_number = 100
    recommendation_user_number = r.check_user_number(recommendation_user_number)
    recommendation_list_length = length
    #print('Available user number: ', recommendation_user_number)

    if random_recommendation:
        rand_n = []  # number of hit products
        rand_t = []  # number of total consumed products
        rand_r = []  # hit ratio
        for j in range(100):
            r.recommendation_list = []
            for i in range(recommendation_user_number):
                r.generate_recommendation_random(user=i, recommendation_list_length=recommendation_list_length)
            n, t, ra = r.hit_ratio_random(mode='Random', recommendation_user_number=recommendation_user_number)
            rand_n.append(n)
            rand_t.append(t)
            rand_r.append(ra)
        print('Random:\t' + str(np.average(rand_n)) + '\t' + str(np.average(rand_t)) + '\t' +
              str(np.average(rand_r)), end='\n', flush=True)

    r.recommendation_list = []
    for i in range(recommendation_user_number):
        r.generate_recommendation(user=i, recommendation_list_length=recommendation_list_length, r=True)
    # print('with rating ', end='')
    r.hit_ratio(mode=ssim_mode, recommendation_user_number=recommendation_user_number)

    '''r.recommendation_list = []
    for i in range(recommendation_user_number):
        r.generate_recommendation(user=i, recommendation_list_length=recommendation_list_length, r=False)
    print('without rating ', end='')
    r.hit_ratio(mode=ssim_mode, recommendation_user_number=recommendation_user_number)'''


category = ['Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
            'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
            'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
            'Kindle_Store_5', 'Movies_and_TV_5', 'Musical_Instruments_5', 'Pet_Supplies_5',
            'Patio_Lawn_and_Garden', 'Sports_and_Outdoors_5',
            'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']
modes = ['ssim', 'cwssim']

length = 50

for prefix in category:
    print(prefix.replace('_5', ''))
    random_recommendation = True
    sp = True
    for ssim_mode in modes:
        recommend(prefix, ssim_mode, random_recommendation, sp, length)
        random_recommendation = False
        sp = False
    print('')
print('\n')



