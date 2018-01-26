import torch
from torch import optim, nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

from model import RandomNet

def train(epoch):
    model.train()
    time_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        before = time.time()
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        after = time.time()
        time_sum += after - before
        
        if batch_idx % 200 == 0:
            print(output)
            print(torch.sum((model.hebb_classifier.permanence > 0).type_as(model.hebb_classifier.permanence)))
            print("Time: ", time_sum / (batch_idx + 1))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    print("Testing")
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("=============================================================================")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("=============================================================================")


if __name__ == "__main__":
    model = RandomNet().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss(weight=None, size_average=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist', train=True, download=True, transform=transforms.ToTensor()), batch_size=1, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist', train=False, download=True, transform=transforms.ToTensor()), batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

    for epoch in range(1, 100):
        train(epoch)
        test()
