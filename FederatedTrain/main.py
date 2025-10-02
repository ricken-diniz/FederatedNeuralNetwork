import torch
import argparse
from torchvision.utils import save_image
from modelNet import Net
import torch.optim as optim
from torchvision import datasets, transforms
from model_mnist import train, test
import requests

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # print(use_cuda)
    torch.manual_seed(args.seed)
    use_cuda = False
    device = 'cpu'

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    mnist_trainset = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    train_loader = torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # img_tensor, label = mnist_trainset[0]
    # save_image(img_tensor, "mnist_example.png")

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        ###################################################################
    torch.save(model.state_dict(), "static_model.pt")
    send_model("static_model.pt","http://localhost:5000/uploadModel")

def send_model(file_path, url):
    with open(file_path, "rb") as f:
        file = {
            "file": (file_path, f)
        }
        response = requests.post(url, files=file)

    print(response.status_code, response.text)

        
if __name__ == '__main__':
    #main() creates the model.pth. saveModel takes the model.pth and random input and creates model.pt.
    # saveModel()
    main()