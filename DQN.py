import gym, math, random, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# Created by Hongyang, Aug 16 2017
# run on my Macbook (cpu)


class ReplayMem(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SexyDQN(nn.Module):

    def __init__(self):
        super(SexyDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)   # why 448

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)       # middle of cart


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # into torch order (CHW)
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]

    # convert to torch-style tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)   # FloatTensor, [3, 160, 320]
    check_ = resize(screen)             # FloatTensor, [3, 40, 80]
    return check_.unsqueeze(0).type(Tensor)     # what is unsqueeze


def optimize_model():
    # global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))  # why transpose here
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_state = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_value = model(state_batch).gather(1, action_batch)
    next_state_value = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_value[non_final_mask] = model(non_final_next_state).max(1)[0]
    next_state_value.volatile = False
    expected_q_value = (next_state_value * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp(-1, 1)

    optimizer.step()


def select_action(state):
    global steps_done
    check_ = random.random()
    eps_thres = EPS_END + (EPS_START - EPS_END) * \
                          math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if check_ > eps_thres:

        net_output = model(Variable(state, volatile=True).type(FloatTensor))
        return net_output.data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])


def plot_durations():
    global eps_width
    plt.figure(2)
    plt.clf()
    eps_width_tensor = FloatTensor(eps_width)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    # eps_width = eps_width.numpy()
    plt.plot(eps_width_tensor.numpy())
    eps_width = eps_width_tensor.numpy().tolist()

    if len(eps_width) >= 100:       # no idea what us going on here
        means = eps_width_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

env = gym.make('CartPole-v0').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

##
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

###
resize = T.Compose([T.ToPILImage(), T.Scale(40, interpolation=Image.CUBIC), T.ToTensor()])
screen_width = 600

# env.reset()
# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
# plt.title('example extracted screen')
# plt.show()


# Start of the main code
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
num_eps = 200

model = SexyDQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMem(10000)
steps_done = 0
eps_width = []


# Main training loop here
# last_sync = 0
for eps_iter in range(num_eps):

    # initialize the environment, state
    env.reset()
    last_screen = get_screen()
    curr_screen = get_screen()
    state = curr_screen - last_screen
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action[0, 0])
        reward = Tensor([reward])

        last_screen = curr_screen
        curr_screen = get_screen()
        if not done:
            next_state = curr_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()
        if done:
            eps_width.append(t + 1)
            plot_durations()
            break

print('Complete')
env.render(close=True)
env.close()
plt.ioff()
plt.show()
