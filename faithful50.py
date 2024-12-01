import torch
import numpy as np
import quantus
from resnet import ResNetAudio  # Assuming ResNetAudio is defined in resnet.py
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') 

# Initializing the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_faithfulness(model, x_batch, y_batch, a_batch, device):
    faithfulness = quantus.MonotonicityCorrelation(
        nr_samples=10,
        features_in_step=2,
        perturb_baseline="uniform",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
    )(model=model, 
      x_batch=x_batch,
      y_batch=y_batch,
      a_batch=a_batch,
      device=device)
            
    return faithfulness

def calculate_uncertainty(model, x_batch, device, num_samples=10):
    model.eval()
    x_batch = torch.from_numpy(x_batch).to(device)
    uncertainties = []

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(x_batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            uncertainties.append(probs.cpu().numpy())
    
    uncertainties = np.array(uncertainties)
    mean_uncertainty = uncertainties.mean(axis=0)
    std_uncertainty = uncertainties.std(axis=0)
    
    return mean_uncertainty, std_uncertainty


def plot_faithfulness_scores(scores):
    print("Plotting faithfulness scores...")  # Debug print
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores, fill=True, bw_adjust=0.5)
    plt.title('Density Plot of Faithfulness Scores for 50 Samples from ResNet18')
    plt.xlabel('Faithfulness Score')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig("faithfulness_density18.png")
    print("Plot saved as faithfulness_density.png")
    plt.show()


def main():
    # Load the pre-computed attribution maps
    sample_data = np.load("Explanations18_batch.npy", allow_pickle=True).item()

    # Access the individual images
    a_batch = sample_data['explanation']
    x_batch = sample_data['mfcc']
    y_batch = sample_data['ground_truth']

    print(x_batch.shape)
    print(a_batch.shape)
    print(y_batch.shape)


    x_batch = np.expand_dims(x_batch, axis=0)
    x_batch = np.transpose(x_batch, (0, 3, 1, 2))
    x_batch = np.transpose(x_batch, (2, 0, 3, 1)) # -> 50, 1, 13, 400
    a_batch = np.expand_dims(a_batch, axis=1)

    
    print("Shape of x_batch after:", x_batch.shape)
    print("Shape of a_batch after:", a_batch.shape)
    print(y_batch)

    
    # Convert to NumPy array since otherwise it gives the error int object not iterable
    print("Type of y_batch:", type(y_batch))



    if not isinstance(y_batch, np.ndarray):
        y_batch = np.array([y_batch])
    print("Type of y_batch after conversion:", type(y_batch))
    print("Shape of y_batch after conversion:", y_batch.shape)

    # Load your own pre-trained model
    model = ResNetAudio(num_classes=10)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("cnn_model.pth"))
    model.eval()
    model = model.to(device)
    
    # Calculate faithfulness
    faithfulness_scores = calculate_faithfulness(model, x_batch, y_batch, a_batch, device)
    print("Faithfulness Scores:", faithfulness_scores)


    sample_indices = np.arange(len(faithfulness_scores))

    # Plot faithfulness scores
    plot_faithfulness_scores(faithfulness_scores)

if __name__ == "__main__":
    main()
