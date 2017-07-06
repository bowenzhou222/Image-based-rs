import numpy as np
import ssim.ssimlib as pyssim
from multiprocessing import Process, Manager, freeze_support, Pool, Lock, Array
import csv
import os
from ssim.utils import get_gaussian_kernel
import time
from itertools import combinations


'''gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

img1 = 'Musical_Instruments_5_image\B0002E2EOE.jpg'
img2 = 'Musical_Instruments_5_image\B0002E2GMY.jpg'
size = None
ssim = pyssim.SSIM(img1, gaussian_kernel_1d, size=None)
print('%.7g' % ssim.ssim_value(img2))
#print(pyssim.SSIM(img1).cw_ssim_value(img2))

'''


def similarity(prefix, img1, img2, arr, mode, lock):
    img1_path = 'subset\subset_images\\' + img1 + '.jpg'
    img2_path = 'subset\subset_images\\' + img2 + '.jpg'
    if mode == 'ssim':
        gaussian_kernel_sigma = 1.5
        gaussian_kernel_width = 11
        gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
        size = None
        ssim = pyssim.SSIM(img1_path, gaussian_kernel_1d, size=size)
        sim = ssim.ssim_value(img2_path)
    else:
        sim = pyssim.SSIM(img1_path).cw_ssim_value(img2_path)
    with lock:
        arr.append([img1, img2, sim])


def sim_imgs(prefix, img1_list, img2_list, arr, mode, lock):
    # img2_list is the entire list
    # print('adding {}...'.format(os.getpid()))
    for i1 in img1_list:
        for i2 in img2_list[img2_list.index(i1)+1:]:
            similarity(prefix, i1, i2, arr, mode, lock)


def save_asin(prefix):
    asin_list = []
    f = open('subset\\'+prefix + '_sub_dense_url.txt', 'r')
    for l in f:
        asin_list.append(l.split(',')[0])
    f.close()
    return asin_list


def xrange(x):
    return iter(range(x))


def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def deduplicate_sim_arr(a):
    a_d = list()
    for e in a:
        exist = 0
        for e_d in a_d:
            if (e_d[0] in e) and (e_d[1] in e):
                exist = 1
                break
            if e[0] == e[1]:
                exist = 1
                break
        if exist == 0:
            a_d.append(e)
    return a_d


def eval_consumption(prefix):
    user_list = []

    csvfile = open('subset\\' + prefix + '_sub_dense.csv', newline='')
    csv_lines = list(csv.reader(csvfile, delimiter=','))
    for line in csv_lines:
        if not line:
            continue
        user_id = str(line[0])
        user_list.append(user_id)
    user_set = set(user_list)  # save all user id into set
    csv_file = open('subset\\' + prefix + '_consumption.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    for user in user_set:
        consumption = [user]
        for line in csv_lines:
            if not line:
                continue
            if str(line[0]) == user:
                consumption.append(str(line[1]))
        csv_writer.writerow(consumption)

    csvfile.close()
    csv_file.close()


def eval_consumption_and_similarity(prefix):
    user_list = []
    csvfile = open('subset\\' + prefix + '_sub_dense.csv', newline='')
    csv_lines = list(csv.reader(csvfile, delimiter=','))
    for line in csv_lines:
        if not line:
            continue
        user_id = str(line[0])
        user_list.append(user_id)
    user_set = set(user_list)  # save all user id into set
    consumption_all_user = []
    for user in user_set:
        consumption = [] # list of consumed products
        for line in csv_lines:
            if not line:
                continue
            if str(line[0]) == user:
                consumption.append(str(line[1]))
        consumption_all_user.append(consumption)
    #eval_average_similarity(prefix, consumption, avg_sim, similarity)
    csvfile.close()
    #print(np.average(similarity))
    #print(len(consumption_all_user))
    return consumption_all_user


def eval_average_similarity(prefix, asin_arr, avg_sim, similarity, lock, raw_lines, lines):
    for sub_asin_arr in asin_arr:
        pairs = combinations(sub_asin_arr, 2)

        #lines = map(lambda x: x[0]+','+x[1], lines)
        #print(len(lines))
        #similarity = []
        for pair in pairs:
            # pair is tuple of form ('aaa', 'bbb')
            #print(pair)
            index = [raw_lines.index(l) for l in raw_lines if pair[0] in l and pair[1] in l]
            if len(index) > 0:
                sim = float(lines[index[0]][2])
            else:
                sim = avg_sim
            with lock:
                similarity.append(sim)

        #print(np.average(similarity))


def global_average_similarity(prefix, mode):
    f = open('subset\image_similarity\\'+prefix+'_'+mode+'.csv', 'r')
    lines = list(csv.reader(f, delimiter=','))
    similarity = []
    for line in lines:
        similarity.append(float(line[2]))
    f.close()
    return np.average(similarity)


if __name__ == '__main__':
    freeze_support()
    processes = 16
    '''category = ['Automotive_5', 'Digital_Music_5', 'Grocery_and_Gourmet_Food_5',
                'Musical_Instruments_5',
                'Office_Products_5', 'Patio_Lawn_and_Garden_5', 'Pet_Supplies_5', 'Tools_and_Home_Improvement_5']'''
    category = ['Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
                'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
                'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
                'Kindle_Store_5', 'Movies_and_TV_5', 'Pet_Supplies_5', 'Sports_and_Outdoors_5',
                'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']
    category_p = ['Digital_Music_5',
                'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
                'Kindle_Store_5', 'Movies_and_TV_5', 'Pet_Supplies_5', 'Sports_and_Outdoors_5',
                'Tools_and_Home_Improvement_5', 'Toys_and_Games_5', 'Video_Games_5']
    category_supplementary = [x for x in category if x not in category_p]
    # prefix = 'Musical_Instruments_5'

    modes = ['cwssim']
    # mode = 'ssim'

    for mode in modes:
        for prefix in category_sup:
            asin_list = []
            asin_list = save_asin(prefix)  # save asin of products into list asin_list
            asin_list_split = split(asin_list, processes)
            #print(len(asin_list_split))
            lock = Lock()
            # arr contains similarity between pairs of images in asin_list. Form is [[img1, img2, sim], [img1, img3, sim]...]
            arr_sim = Manager().list()
            # print(arr_sim)
            threads = []
            for i in range(processes):
                #print('creating thread...')
                t = Process(target=sim_imgs, args=(prefix, asin_list_split[i], asin_list, arr_sim, mode, lock, ))
                t.daemon = True
                threads.append(t)

            #print('{} threads'.format(len(threads)))
            start_time = time.time()

            for j in range(len(threads)):
                threads[j].start()

            for k in range(len(threads)):
                threads[k].join()

            # print('CW-SSIM finished')
            arr_sim = list(arr_sim)
            # print(len(arr_sim))
            #arr_sim_d = deduplicate_sim_arr(arr_sim)
            #print(arr_sim_d)
            #print(len(arr_sim_d))
            with open('subset\image_similarity\\'+prefix+'_'+mode+'.csv', 'w', newline='') as f:
                #print('start writing to csv')
                writer = csv.writer(f)
                writer.writerows(arr_sim)
                #print('finish writing to csv')
            result_file = open(r'subset\result.txt', 'a')  # number of pairs and operation time
            # print('Time consumed: {}'.format(time.time() - start_time))
            result_file.write('\t'.join((prefix, mode, str(len(arr_sim)), str(time.time() - start_time), '\n')))
            result_file.close()






    for mode in modes:
        for prefix in category_p:
            lock = Lock()
            mode = 'cwssim'
            # print(mode)
            avg_sim = global_average_similarity(prefix, mode)
            # print(avg)
            consumption_each_user = eval_consumption_and_similarity(prefix)

            similarity = Manager().list()
            consumption_each_user_slice = split(consumption_each_user, processes)

            f = open('subset\image_similarity\\' + prefix + '_'+mode+'.csv', 'r')
            raw_lines = f.readlines()
            f.close()
            f = open('subset\image_similarity\\' + prefix + '_'+mode+'.csv', 'r')
            lines = list(csv.reader(f, delimiter=','))
            f.close()

            threads = []
            for i in range(processes):
                # print('creating thread...')
                t = Process(target=eval_average_similarity,
                            args=(prefix, consumption_each_user_slice[i], avg_sim, similarity, lock, raw_lines, lines, ))
                t.daemon = True
                threads.append(t)

            # print('{} threads'.format(len(threads)))

            for j in range(len(threads)):
                threads[j].start()

            for k in range(len(threads)):
                threads[k].join()

            similarity_result_file = open('subset\consumed_product_similarity_result.txt', 'a')
            #
            similarity_result_file.write('\t'.join((prefix, mode, str(avg_sim), str(np.average(similarity)))))
            # print('Average similarity: {}'.format(np.average(similarity)))
            similarity_result_file.close()


'''def testFunc(sum1, cc, lock):
    with lock:
        sum1.value += cc

if __name__ == '__main__':
    freeze_support()
    lock = Lock()
    manager = Manager()
    sum1 = manager.Value('tmp', 0)
    threads = []

    for ll in range(100):
        t = Process(target=testFunc, args=(sum1, 1, lock,))
        t.daemon = True
        threads.append(t)

    for i in range(len(threads)):
        threads[i].start()

    for j in range(len(threads)):
        threads[j].join()

    print("------------------------")
    print('process id:', os.getpid())
    print(sum1.value)'''

