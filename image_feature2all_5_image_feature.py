# extract image features of products in prefix_5.csv datasets
import struct
import numpy as np
from urllib import request
# from matplotlib import pylab as plt
# from PIL import Image
# from scipy import misc
import json
import csv

class Rating:
    def __init__(self, rating_file):
        self.rating_file = rating_file
        self.count()

    def count(self):
        self.user_vector = list()
        self.item_vector = list()
        self.rating_vector = list()

        with open(self.rating_file, newline='') as csvfile:
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
        #self.mapping()

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
        self.item_vector = list(map(lambda x: [x, self.item_id_dictionary[x]], self.item_vector))


def readImageFeatures(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if asin == '':
            break
        feature = []
        for i in range(4096):
            #try:
            feature.append(struct.unpack('f', f.read(4)))
            #except struct.error:
                #feature.append((0,))
        yield asin, feature


category = ['Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
            'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
            'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
            'Kindle_Store_5', 'Movies_and_TV_5', 'Musical_Instruments_5', 'Office_Products_5', 'Pet_Supplies_5',
            'Patio_Lawn_and_Garden', 'Sports_and_Outdoors_5',
            'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']

category_small = ['Musical_Instruments_5', 'Patio_Lawn_and_Garden_5', 'Automotive_5', 'Office_Products_5',
                  'Digital_Music_5', 'Baby_5', 'Pet_Supplies_5', 'Grocery_and_Gourmet_Food_5', 'Tools_and_Home_Improvement_5']

category_small_small = ['Musical_Instruments_5', 'Patio_Lawn_and_Garden_5', 'Automotive_5', 'Office_Products_5',
                  'Digital_Music_5', 'Pet_Supplies_5', 'Grocery_and_Gourmet_Food_5', 'Tools_and_Home_Improvement_5']

prefix = 'Office_Products_5'

rating_file_path = r'd:\Study\research\amazon\experiment\json data\subset\\'

rating_file = rating_file_path+prefix+'_sub_dense.csv'

r = Rating(rating_file)

fw = open('All_5_image_features\image_features_'+prefix+'_sub_dense.json', 'w')
n = 0
m = 0
for asin, image in readImageFeatures('image_features\image_features_'+prefix.replace('_5', '')+'.b'):
    n += 1
    if n-m*5000>=5000:
        m+=1
        print(m*5000)
    data = {}
    asin = asin.decode('ascii')
    if asin in list(r.item_id_set):
        image = [img[0] for img in image]
        data['asin'] = asin
        data['image'] = image
        fw.write(json.dumps(data)+'\n')
    # print(asin.decode('ascii'), [img[0] for img in image])
    # print(type(image[0]))
    '''for i in range(len(image)):
        if isinstance(image[i], tuple) == False:
            print('not tuple')
        if len(image[i]) != 1:
            print('length of tuple is not 1')'''
print(n)
fw.close()
