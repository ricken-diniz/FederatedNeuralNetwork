import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def createTracedModel(model, random_input):
    traced_net = torch.jit.trace(model,random_input)
    traced_net.save("model_trace.pt")

    print("Success - model_trace was saved!")


def saveModel():
    use_cuda = False
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    train_image, train_target= mnist_testset[24]
    train_image.show()

    device = 'cpu'
    model = torch.load("model.pth").to(device)
    loader = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    tensor_image = loader(train_image).unsqueeze(0).to(device)
    output = model(tensor_image)
    pred = output.max(1, keepdim=True)[1]
    pred = torch.squeeze(pred)

    print("Success - Train target: " + str(train_target.cpu().numpy()) + " Prediction: " + str(pred.cpu().numpy()))


    # TRACING THE MODEL comment out if you dont wanna save the trace model in a demo run
    createTracedModel(model, tensor_image)


