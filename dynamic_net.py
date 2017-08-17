import numpy as np
import torch
from torch.autograd import Variable
import random


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
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
model = DynamicNet(D_in, H, D_out)

learning_rate = 1e-6
# loss_fn = torch.nn.MSELoss(size_average=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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
