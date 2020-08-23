import torch
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from supernet.model import SuperNet
from supernet.train import train, test


def main():
    # Training settings
    seed = 1
    batch_size = 64
    test_batch_size = 1000
    epochs = 14
    lr = 1.0
    gamma = 0.7
    no_cuda = True
    dry_run = False
    log_interval = 50
    save_model = False
    model = SuperNet()

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
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    test_acc = []

    if subnet is not None:
        model.set_choice(subnet)

    for epoch in range(1, epochs + 1):
        train_super(model, device, train_loader, optimizer, epoch, log_interval, dry_run, subnet)

        test_acc_ = []
        if subnet == None:
            for choice in [[0, 0], [1, 0], [0, 1], [1, 1]]:
                model.set_choice(choice)
                test_acc_.append(test_super(model, device, test_loader))
            test_acc.append(test_acc_)
        else:
            model.set_choice(subnet)
            test_acc.append(test_super(model, device, test_loader))

        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_supernet.pt")

    return test_acc


if __name__ == '__main__':
    main()
