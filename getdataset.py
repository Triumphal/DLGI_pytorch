from torch.utils import data
from torchvision import transforms
from skimage import io
import os
import matplotlib.pyplot as plt
import numpy as np
import torch



def get_images_and_labels(images_path, labels_path):
    image_list = []
    label_list = []
    for i in range(9000):
        image_path = images_path + '/' +str(i+1)+'.tif'
        label_path = labels_path + '/' +str(i+1)+'.bmp'
        image_list.append(io.imread(image_path)[np.newaxis,:].astype(np.float32))
        label_list.append(io.imread(label_path)[np.newaxis,:].astype(np.float32))
    
    return np.array(image_list), np.array(label_list)



class MyDataset(data.Dataset):
    def __init__(self, images_path, labels_path) -> None:
        super().__init__()
        self.images, self.labels = get_images_and_labels(images_path,labels_path)

    def __getitem__(self, index):
        # 输入的图像归一化处理
        image = torch.Tensor(self.images[index])
        label = torch.Tensor(self.labels[index])
        image_mean = torch.sqrt(torch.sum(torch.square(image - torch.mean(image)))/64)
        image = (image - torch.mean(image))/image_mean

        return image, label
    
    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    images_path = r"./data/Inputs_mnist_train_64"
    labels_path = r"./data/Labels_mnist_train"

    mydata = MyDataset(images_path,labels_path)

    train_iter = data.DataLoader(mydata, batch_size=5, shuffle=True)
    for i, (X, Y) in  enumerate(train_iter):
        print(X.shape, Y.shape)
        break
    # X, Y = next(iter(train_iter))