import os
import random

path = "E:/Desktop/CASIA-WebFace/CASIA-WebFace"
train = "E:/Desktop/CASIA-WebFace/train"
verify = "E:/Desktop/CASIA-WebFace/verify"
test = "E:/Desktop/CASIA-WebFace/test"


def select_img(path, new_path, count):
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    for index, dirname in enumerate(os.listdir(path)):
        img_dir = os.path.join(path, dirname)

        img_list = os.listdir(img_dir)
        if len(img_list) > count:
            new_img_dir = os.path.join(new_path, dirname)
            os.makedirs(new_img_dir)

            for item in img_list:
                img_path = os.path.join(img_dir, item)
                new_img_path = os.path.join(new_img_dir, item)

                with open(img_path, mode='rb') as f1:
                    with open(new_img_path, mode='wb') as f2:
                        data = f1.read()
                        f2.write(data)


def calculate(title, path):
    count = 0
    for item in os.listdir(path):
        count += len(os.listdir(os.path.join(path, item)))
    print(title, count)


def split_dataset(path, new_path, scale):
    img_dirname_list = os.listdir(path)

    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    for img_dirname in img_dirname_list:
        # 图片文件夹路径
        new_img_dir = os.path.join(new_path, img_dirname)
        img_dir = os.path.join(path, img_dirname)

        if not os.path.isdir(new_img_dir):
            os.makedirs(new_img_dir)

        # 图片列表
        img_name_list = os.listdir(img_dir)
        # 图片数量
        img_count = len(img_name_list)
        # 切割的数量
        split_num = int(img_count * scale)
        # 随机打乱
        random.shuffle(img_name_list)

        for i in range(split_num):
            print(f'\r进度：{i + 1}/{split_num}', end="")
            # 图片地址
            img_path = os.path.join(img_dir, img_name_list[i])
            new_img_path = os.path.join(new_img_dir, img_name_list[i])
            with open(img_path, mode="rb") as f1:
                with open(new_img_path, mode="wb") as f2:
                    data = f1.read()
                    f2.write(data)
            os.remove(img_path)
    print("")


if __name__ == "__main__":
    select_img(path, train, 200)
    calculate("选取后的图片：", train)
    split_dataset(train, verify, 0.1)
    calculate("训练集+验证集：", verify)
    split_dataset(verify, test, 0.5)
    calculate("最后输出训练数据集：", train)
    calculate("最后输出验证数据集：", verify)
    calculate("最后输出验证数据集：", test)
