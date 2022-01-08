#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('hot_dog', 'not_hot_dog')
BATCH_SIZE = 10
ROOT_PATH = './dataset/'

def load_data(root_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data = datasets.ImageFolder(root=root_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    
    return data_loader

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    loader = load_data(ROOT_PATH, BATCH_SIZE)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % CLASSES[labels[j]] for j in range(BATCH_SIZE)))
    

if __name__ == '__main__':
    main()