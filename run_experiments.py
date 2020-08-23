import numpy as np
import matplotlib.pyplot as plt
from supernet.model import SuperNet
from supernet.train import train_supernet_mnist
import pickle


def main():
    # Training settings
    training_settings = {'seed': 1,
                         'batch_size': 64,
                         'test_batch_size': 1000,
                         'epochs': 14,
                         'learning_rate': 1.0,
                         'gamma': 0.7,
                         'no_cuda': True,
                         'log_interval': 50,
                         'save_model': True
                         }

    # 1. Train SuperNet and evaluate top-1 accuracy of sampled nets

    top1_oneshot_ = np.array(train_supernet_mnist(SuperNet(), training_settings))
    with open('top1_oneshot_.pickle', 'wb') as handle:
        pickle.dump(top1_oneshot_, handle)

    fig, ax = plt.subplots()
    ax.plot(top1_oneshot_, '-s')
    ax.grid()
    ax.legend([[0, 0], [1, 0], [0, 1], [1, 1]])
    plt.title('One-shot SuperNet training')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 accuracy on test set')
    fig.savefig('figures/top1_oneshot_.png')
    plt.show()

    # 2. Train stand-alone subnets from scratch
    top1_standalone = []
    training_settings['epochs'] = 14
    for k, subnet in enumerate([[0, 0], [1, 0], [0, 1], [1, 1]]):
        top1_standalone.append(train_supernet_mnist(SuperNet(), training_settings, subnet=subnet))
    top1_standalone_ = np.array(top1_standalone)
    with open('top1_standalone_.pickle', 'wb') as handle:
        pickle.dump(top1_standalone_, handle)

    fig, ax = plt.subplots()
    ax.plot(top1_standalone_.T, '-s')
    ax.grid()
    ax.legend([[0, 0], [1, 0], [0, 1], [1, 1]])
    plt.title('Stand-alone model training')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 accuracy on test set')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(top1_oneshot_[-1, :], top1_standalone_.T[-1, :], 'o')
    ax.grid()
    plt.xlabel('One-shot model accuracy')
    plt.ylabel('Stand-alone model accuracy')
    fig.savefig('figures/oneshot_v_standalone_.png')
    plt.show()


if __name__ == '__main__':
    main()
