import numpy as np
import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )
model = TwoLayerNet(D_in, H, D_out)

learning_rate = 1e-6
# loss_fn = torch.nn.MSELoss(size_average=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(500):

    y_pred = model(x)

    # loss = loss_fn(y_pred, y)
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # model.zero_grad()
    optimizer.zero_grad()

    loss.backward()

    # for param in model.parameters():
    #     param.data -= learning_rate * param.grad.data
    optimizer.step()
