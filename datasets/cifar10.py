import os
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["train", "test"]:
        out_dir = f"cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("downloading...")

        # 使用Python的tempfile模块创建一个临时目录
        # 该临时目录在with语句块结束时会自动被清理，用于下载CIFAR-10数据集的临时存储
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            # 获取数据集中的第i个样本，其中image是一个PIL（Python Imaging Library）图像对象
            # label是该样本的类别标签，它是一个0到9之间的整数，对应于CLASSES中定义的类别名称
            image, label = dataset[i]

            # 件名的格式为类别名称_序号.png，其中CLASSES[label]根据类别标签label查找对应的类别名称
            # :05d表示将序号格式化为5位数字（不足5位时在前面补0）
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)


if __name__ == "__main__":
    main()
