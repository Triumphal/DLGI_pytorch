from getdataset import MyDataset
from dlginet import FirstNet, SecondNet, LastNet, ResBlock
from torch import nn
from torch.utils import data
import torch,time
from display_utils import  try_gpu


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.trunc_normal_(m.weight,std=0.1)
        # nn.init.normal_(m.weight, std=0.1)

def train(net, train_iter, num_epochs, lr, device, init_weight=True):
    """使用GPU进行训练"""
    if init_weight:# 网络重新初始化
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if device.index == 0: # 开启cudnn加速
        print("cudnn is available")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        net.train()
        begin = time.time()
        for i, (X, Y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, Y = X.to(device), Y.to(device)
            Y_hat = net(X)
            l = loss(Y_hat, Y)
            l.backward()
            optimizer.step()
        end = time.time()
        if epoch%1000 == 0:
            torch.save(net,'./net/net_'+str(epoch)+'.pth')
        print(f"[epoch {epoch+1}]: training loss: {l:f}, consuming time:{end-begin:.4f} s")

if __name__ == '__main__':
    net = nn.Sequential(FirstNet(64,32),SecondNet(32,32),LastNet(128,1)) #,nn.Conv2d(128,1,kernel_size=2,stride=2)
    bath_size = 5
    data_path = r"./data/Inputs_mnist_train_64"
    label_path = r"./data/Labels_mnist_train"
    mydata = MyDataset(data_path,label_path)
    train_iter = data.DataLoader(mydata, batch_size=bath_size, shuffle=True)
    lr, num_epochs = 1e-2, 500
    train(net, train_iter, num_epochs, lr, try_gpu())