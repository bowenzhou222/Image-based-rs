import urllib.request
from multiprocessing import Pool, freeze_support
import multiprocessing


def save_pic(image_url, image_path):
    try:
        r = urllib.request.urlopen(image_url)
        image = r.read()
    except:
        image = b''
    with open(image_path, 'wb')as image_file:
        image_file.write(image)


def crawl(prefix):
    url_list = list()
    image_path_list = list()
    f = open('subset\\' + prefix + '_sub_dense_url.txt', 'r')
    for l in f:
        my_str = l.split(',')
        image_name = my_str[0] + '.jpg'
        url = ''.join(my_str[1:])
        image_path_list.append('subset\subset_images\\'+image_name)
        url_list.append(url)
    f.close()
    for i in range(len(image_path_list)):
        save_pic(url_list[i], image_path_list[i])


def split_targets(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == '__main__':
    freeze_support()

    category = ['Amazon_Instant_Video_5', 'Apps_for_Android_5', 'Automotive_5', 'Baby_5', 'Beauty_5', 'Books_5',
                'CDs_and_Vinyl_5', 'Cell_Phones_and_Accessories_5', 'Clothing_Shoes_and_Jewelry_5', 'Digital_Music_5',
                'Electronics_5', 'Grocery_and_Gourmet_Food_5', 'Health_and_Personal_Care_5', 'Home_and_Kitchen_5',
                'Kindle_Store_5', 'Movies_and_TV_5', 'Musical_Instruments_5', 'Office_Products_5',
                'Patio_Lawn_and_Garden_5',
                'Pet_Supplies_5', 'Sports_and_Outdoors_5', 'Tools_and_Home_Improvement_5', 'Toys_and_Games_5',
                'Video_Games_5']
    category_slice = split_targets(category, 3)

    for c in category_slice:
        pool = multiprocessing.Pool(processes=8)
        pool.map(crawl, c)
        pool.close()
        pool.join()
