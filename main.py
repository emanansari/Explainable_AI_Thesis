import dataset as D
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from resnet import ResNetAudio
from train import train, test
from pytorch_grad_cam import GradCAM

DATAPATH = '/home3/s4317394/pytorch/audio'
FOLDS_TRAIN = [i for i in range(1, 10)]
FOLDS_TEST = [10]
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
target_sr = 44100
n_mfcc = 13
max_length = 400  


def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')


    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=target_sr,
        n_mfcc=n_mfcc,
        melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 100},
    )

    # Instantiating our dataset object and create data loader
    train_dataset = D.UrbanSound8K(DATAPATH, device, folders_in=FOLDS_TRAIN, transform=mfcc_transform)
    torch.save(train_dataset, "transformed_dataset.pth")
    
    test_dataset = D.UrbanSound8K(DATAPATH, device, folders_in=FOLDS_TEST, transform=mfcc_transform)

    print(f"Number of samples in the training dataset: {len(train_dataset)}")

    train_dataloader = create_data_loader(train_dataset, BATCH_SIZE)
    test_dataloader = create_data_loader(test_dataset, BATCH_SIZE)

    # Construct model and assign it to device
    cnn = ResNetAudio(num_classes=10).to(device)

    print(cnn)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(cnn.parameters(), lr=LEARNING_RATE)

    # Train model
    train(cnn, train_dataloader, BATCH_SIZE, optimiser, loss_fn, EPOCHS, device)

    # Save model
    torch.save(cnn.state_dict(), "cnn_model.pth")
    print("Trained CNN model saved at cnn_model.pth")

    # Test the model
    test_loss, test_accuracy = test(cnn, test_dataloader, loss_fn, device='cpu')