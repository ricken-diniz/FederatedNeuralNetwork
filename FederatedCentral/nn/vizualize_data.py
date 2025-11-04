import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from modelNet import Net
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    model = Net()
    model.load_state_dict(torch.load('federated_model.pt'))
    model.eval()

    mnist_trainset = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True, **{})

    predicts = {
        0: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        1: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        2: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        3: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        4: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        5: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        6: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        7: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        8: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        9: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        }
    }

    with torch.no_grad():

        for data, target in mnist_trainset:
            data, target = data.to('cpu'), target.to('cpu')
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1, keepdim=True)
            predicted_class = predicted[0].item()
            predicts[target[0].item()][predicted_class] += 1

    df = pd.DataFrame(predicts).T
    print(df)
    plt.figure(figsize=(10, 7))
    plt.title("Matriz de Confusão")
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig("analisys/confusion_matrix_before_poisoning.png")
    plt.show()
    print("Matriz de confusão salva como './analisys/confusion_matrix.png'")

if __name__ == '__main__':
    main()
