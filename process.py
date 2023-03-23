import os
import random
from natsort import natsorted
import numpy as np
from PIL import Image

np.random.seed(42)

def rgb2gray(path, save_path):
    rgb = Image.open(path)
    gray = rgb.convert('L')
    rgb.save(save_path.replace("gray","rgb"))
    gray.save(save_path)
    print(f"gray image saved in {save_path}")
    return

def read_txt2list(path):
    with open(path, 'r') as fp:
        filelist = fp.readlines()
    ret_l = [item[0:-1] for item in list(filelist) if len(item.strip())>0]
    return ret_l

def listdir(path, list_name): # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            if '.jpg' in file_path:
                list_name.append(file_path)

def write_list2txt(list, path):
    fp=open(path,'w')
    fp.writelines([item + '\n' for item in list])
    fp.close()


def random_take(tot_list, num):
    i = 0
    sub_list = []
    while i < num:
        rand_id = np.random.randint(0,len(tot_list)-1)
        if tot_list[rand_id] in sub_list:
            continue
        else:
            sub_list.append(tot_list[rand_id])
            i += 1
    return sub_list


if __name__ == '__main__':
    train_root = '/archive/xiaopeng/train_256_places365standard/'
    test_root = '/archive/xiaopeng/test_256/'

    tot_train_txt = os.path.join(train_root,"tot_train_filelist.txt")
    tot_test_txt = os.path.join(test_root,"tot_test_filelist.txt")

    sub_train_txt = os.path.join(train_root,"sub_train_filelist.txt")
    sub_test_txt = os.path.join(test_root,"sub_test_filelist.txt")

    sub_train_dir_gray = os.path.join(train_root,"sub_train_gray")
    sub_test_dir_gray = os.path.join(test_root,"sub_test_gray")

    sub_train_dir_rgb = os.path.join(train_root,"sub_train_rgb")
    sub_test_dir_rgb = os.path.join(test_root,"sub_test_rgb")

    os.makedirs(sub_train_dir_gray, exist_ok=True)
    os.makedirs(sub_test_dir_gray, exist_ok=True)

    os.makedirs(sub_train_dir_rgb, exist_ok=True)
    os.makedirs(sub_test_dir_rgb, exist_ok=True)

    tot_train = []
    tot_test = []

    sub_train = []
    sub_test = []
    if not os.path.exists(sub_train_txt):
        if not os.path.exists(tot_train_txt):
            listdir(train_root, tot_train)
            listdir(test_root, tot_test)

            tot_train = natsorted(tot_train)
            tot_test = natsorted(tot_test)
            write_list2txt(tot_train, tot_train_txt)
            write_list2txt(tot_test, tot_test_txt)
        else:
            tot_train = read_txt2list(tot_train_txt)
            tot_test = read_txt2list(tot_test_txt)

            print(f"len of tot_train={len(tot_train)}")
            print(f"len of tot_test={len(tot_test)}")

            train_num = 50000
            test_num = 10000

            sub_train = random_take(tot_train, 50000)
            sub_test = random_take(tot_test, 10000)

            print(f"len of sub_train={len(sub_train)}")
            print(f"len of sub_test={len(sub_test)}")

            write_list2txt(sub_train, sub_train_txt)
            write_list2txt(sub_test, sub_test_txt)
        print("done")
    else:
        sub_train = read_txt2list(sub_train_txt)
        sub_test = read_txt2list(sub_test_txt)

        for item in sub_train:
            l = item.split("/")
            name = l[-3]+l[-2]+l[-1]
            rgb2gray(item, os.path.join(sub_train_dir_gray, name))

        for item in sub_test:
            l = item.split("/")
            name = l[-1]
            rgb2gray(item, os.path.join(sub_test_dir_gray, name))


