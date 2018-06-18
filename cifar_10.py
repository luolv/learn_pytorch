import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
show = ToPILImage()

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #归一化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='f://cifar10/', train=True, download=True, transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=4)

# 测试集
testset = tv.datasets.CIFAR10(
    'f://cifar10/', train=False, download=True, transform=transform)

testloader = t.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

data, label = trainset[100]
print(classes[label])
npimg = data.numpy()
npimg = (npimg + 1) / 2

plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()
