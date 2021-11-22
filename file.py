import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
print(x2)
# 存储一个张量列表，然后把他们读回内存
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)
# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
# 实例一个模型，然后加载他的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
# model.eval()的作用是不启用 Batch Normalization 和 Dropout,测试模型用
print(clone.eval())# print(clone)也是一样的
Y_clone = clone(X)
print(Y_clone == Y)