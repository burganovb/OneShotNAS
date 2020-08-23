import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def train(model, device, train_loader, optimizer, epoch, log_interval, loss_func, subnet=None):
    if subnet is not None:
        model.set_subnet(subnet)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if subnet is None:
            model.sample_subnet()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def train_supernet_mnist(nnet, training_settings, subnet=None, test_all=False):
    # Training settings
    seed = training_settings['seed']
    batch_size = training_settings['batch_size']
    test_batch_size = training_settings['test_batch_size']
    epochs = training_settings['epochs']
    lr = training_settings['learning_rate']
    gamma = training_settings['gamma']
    no_cuda = training_settings['no_cuda']
    log_interval = training_settings['log_interval']
    save_model = training_settings['save_model']

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size)

    model = nnet.to(device)
    if subnet is not None:
        model.set_subnet(subnet)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    test_acc = []

    if subnet is None:
        print("\nTraining SuperNet\n")
    else:
        print("\nTraining subnet {}\n".format(subnet))

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, F.nll_loss, subnet)

        test_acc_ = []
        if subnet is None or test_all:
            for choice in [[0, 0], [1, 0], [0, 1], [1, 1]]:
                model.set_subnet(choice)
                test_acc_.append(test(model, device, test_loader, F.nll_loss))
            test_acc.append(test_acc_)
        else:
            test_acc.append(test(model, device, test_loader, F.nll_loss))

        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_supernet.pt")

    return test_acc
