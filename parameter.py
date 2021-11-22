import torch
from torch import nn

# 参数管理

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4)) # 或者torch.rand(2, 4)
out = net(X)
print(out)

# 参数访问，把权重拿出来
print(net[2].state_dict()) # net[2]拿到的就是最后一个输出层(8, 1)
print(net[2].bias)
print(net[2].bias.data)
print(type(net[2].bias))
print(net[2].weight.grad == None) # 这里还没有反向计算，所以没有梯度

# 一次性访问所有参数
# *解包，访问list中的每一个值
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)

# 从嵌套块中搜集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())# 和Sequential不同在于，这里可以自定义层的名字
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
out = rgnet(X)
print(out)
print(rgnet)

# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)# 下划线代表原地操作
        nn.init.zeros_(m.bias)

net.apply(init_normal)# apply对net所有的层都调用这个函数
print(net[0].weight.data[0], net[0].bias.data[0])

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)# apply对net所有的层都调用这个函数
print(net[0].weight.data[0], net[0].bias.data[0])

# 对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
# 还可以随便定义
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])

# 直接替换也行
net[0].weight.data[:] += 1

# 参数绑定
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值。
print(net[2].weight.data[0] == net[4].weight.data[0])