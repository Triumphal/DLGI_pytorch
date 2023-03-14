import torch
from torch import nn
from torch.nn import functional as F
# 定义中间网络要使用的残差块
# 定义残差网络
class ResBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): # 这里默认输入和输出的通道数一致
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2) # 默认stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y +=  X 
        return F.relu(Y)
#多个ResBolck组合在一起
def numResBolck(in_chanels, output_channels, num_residuals=4):
    blk =[]
    for _ in range(num_residuals):
        blk.append(ResBlock(in_chanels, output_channels))
    return nn.Sequential(*blk)


# 将网络分成3部分
class FirstNet(nn.Module):
    def __init__(self, in_features, out_features=32,dp_prob=0.1) -> None:
        super().__init__()
        self.ft = nn.Flatten() # 将数据展成一维

        self.fc1 = nn.Linear(in_features, 1024) # pytorch中线性层的偏置是自动开启的
        self.bn1 = nn.BatchNorm1d(1024, eps=0.001, momentum=0.99)
        self.dp1 = nn.Dropout(dp_prob)

        self.fc2 = nn.Linear(1024, 4096)
        self.bn2 = nn.BatchNorm1d(4096, eps=0.001, momentum=0.99)
        self.dp2 = nn.Dropout(dp_prob)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16, eps=0.001, momentum=0.9) 
        self.dp3 = nn.Dropout(dp_prob)

        self.conv2 = nn.Conv2d(16, out_features, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(out_features, eps=0.001, momentum=0.9)
        self.dp4 = nn.Dropout(dp_prob)

    def forward(self, X):
        Y = self.dp1(F.relu(self.bn1(self.fc1(self.ft(X)))))
        Y = self.dp2(F.relu(self.bn2(self.fc2(Y))))
        Y = self.dp3(F.relu(self.bn3(self.conv1(torch.reshape(Y, (-1,1,64, 64))))))
        Y = self.dp4(F.relu(self.bn4(self.conv2(Y))))
        return Y
    
# 定义并行网络，与GoogLNet类似
class SecondNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None: # 输入和输出的通道数一致
        super().__init__()
        # 线路1
        self.p1_1 = numResBolck(in_channels, out_channels)
        # 线路2 下采样2,然后上采样2
        self.p2_1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.p2_2 = numResBolck(in_channels, out_channels)
        self.p2_3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # 线路3 下采样4,然后上采样4
        self.p3_1 = nn.MaxPool2d(kernel_size=4,stride=4)
        self.p3_2 = numResBolck(in_channels, out_channels)
        self.p3_3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.p3_4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # 线路4 下采样8,然后上采样4
        self.p4_1 = nn.MaxPool2d(kernel_size=8,stride=8)
        self.p4_2 = numResBolck(in_channels, out_channels)
        self.p4_3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.p4_4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.p4_5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, X):
        p1 = self.p1_1(X)
        p2 = F.relu(self.p2_3(self.p2_2(self.p2_1(X))))
        p3 = F.relu(self.p3_4(F.relu(self.p3_3(self.p3_2(self.p3_1(X))))))
        p4 = F.relu(self.p4_5(F.relu(self.p4_4(F.relu(self.p4_3(self.p4_2(self.p4_1(X))))))))
        # 在通道维度上连接输出
        return torch.cat((p1, p2, p3, p4), dim=1)
    
class LastNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, dp_prob=0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2t = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

        self.dp1 = nn.Dropout(dp_prob)
        self.dp2 = nn.Dropout(dp_prob)
        self.dp3 = nn.Dropout(dp_prob)
    
    def forward(self, X):
        Y = self.dp1(F.relu(self.bn1(self.conv1(X))))
        Y = self.mp1(Y)
        Y = self.dp2(F.relu(self.bn2(self.conv2(Y))))
        Y = self.mp2(Y)
        Y = self.dp3(F.relu(self.conv2t(Y)))
        return Y

    
if __name__ == "__main__":
    import getdataset
    from torch.utils import data
    def init_weights(m):
        if type(m) == nn.Linear or type == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.1)
    net = nn.Sequential(FirstNet(64,32),SecondNet(32,32),LastNet(128,1)) #

    net.apply(init_weights)

    images_path = r"./data/Inputs_mnist_train_64"
    labels_path = r"./data/Labels_mnist_train"

    mydata = getdataset.MyDataset(images_path, labels_path)
    train_iter = data.DataLoader(mydata, batch_size=5, shuffle=True)
    X,Y  = next(iter(train_iter))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t', X.shape)


    # X,Y = next(iter(train_iter))
    # print(X.shape) 

