import torch
import torchvision
import torchvision.transforms as transforms

# 定义图像预处理管道
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]范围
])

# 加载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt
import numpy as np

# 显示图像的函数
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机获取一些训练图片
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示图片
imshow(torchvision.utils.make_grid(images[:4]))
# 打印标签
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输入通道3(RGB),输出32,卷积核3x3
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # 全连接层
        self.fc2 = nn.Linear(64, 10)  # 输出10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = torch.flatten(x, 1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BasicCNN()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
for epoch in range(10):  # 训练10轮
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()  # 梯度清零
        
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
        
    print(f'Epoch {epoch + 1}, loss: {running_loss / len(trainloader):.3f}')
    # PyTorch数据增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 获取一批测试图片
dataiter = iter(testloader)
images, labels = next(dataiter)

# 预测
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 显示图片和预测结果
imshow(torchvision.utils.make_grid(images[:4]))
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
# 保存
torch.save(model.state_dict(), 'cifar10_model.pth')

# 加载
model = BasicCNN()
model.load_state_dict(torch.load('cifar10_model.pth'))