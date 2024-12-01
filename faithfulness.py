import torch
import numpy as np
import quantus
from resnet import ResNetAudio  # Assuming ResNetAudio is defined in resnet.py
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# Initializing the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def perturb_input(x, step, a, baseline_type="uniform"):
    perturbed_x = x.clone()
    # Implement perturbation logic based on the baseline type
    if baseline_type == "uniform":
        perturbed_x = perturbed_x * (1 - a) + torch.rand_like(perturbed_x) * a
    return perturbed_x

def calculate_performance_impact(model, x, y, a, device, nr_samples=10, features_in_step=1):
    x, y, a = x.to(device), y.to(device), a.to(device)
    model.eval()
    original_output = model(x.unsqueeze(0))
    original_loss = torch.nn.functional.cross_entropy(original_output, y.unsqueeze(0))

    impacts = []
    for step in range(nr_samples):
        perturbed_x = perturb_input(x, step, a, baseline_type="uniform")
        perturbed_output = model(perturbed_x.unsqueeze(0))
        perturbed_loss = torch.nn.functional.cross_entropy(perturbed_output, y.unsqueeze(0))
        
        # Calculate performance impact
        impact = perturbed_loss - original_loss
        impacts.append(impact.item())
    
    return impacts

def plot_performance_impact(impacts):
    steps = range(len(impacts))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, impacts, marker='o', linestyle='-')
    plt.title('Performance Impact Over Perturbation Steps')
    plt.xlabel('Perturbation Step')
    plt.ylabel('Change in Loss')
    plt.grid(True)
    plt.savefig('performance_impact_plot.png')
    plt.close()


def calculate_faithfulness(model, x_batch, y_batch, a_batch, device):
    metric = quantus.MonotonicityCorrelation(
        nr_samples=10,
        features_in_step=1,
        perturb_baseline="uniform",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
    )
    faithfulness = metric(model=model,
                          x_batch=x_batch,
                          y_batch=y_batch,
                          a_batch=a_batch,
                          device=device)
            
    return faithfulness

import numpy as np

def save_single_sample(sample_data):
    # Extract the individual samples
    a_batch = sample_data['explanation']
    x_batch = sample_data['mfcc']
    y_batch = sample_data['ground_truth']

    sample_index = random.randint(0, x_batch.shape[0] - 1)

    # Select the sample at the specified index
    x_sample = x_batch[sample_index]
    a_sample = a_batch[sample_index]
    y_sample = y_batch[sample_index]

    # Create a new dictionary with the single sample
    single_sample_data = {
        'explanation': a_sample,
        'mfcc': x_sample,
        'ground_truth': y_sample
    }

    # Save the single sample to a new .npy file
    np.save("single_sample.npy", single_sample_data)
    print("Single sample saved as single_sample.npy")
    print(sample_index)

def main():
    # Load the pre-computed attribution maps
    sample_data = np.load("Explanations_batch.npy", allow_pickle=True).item()

    # Save a single sample (e.g., the first sample)
    save_single_sample(sample_data)

    # Load the pre-computed attribution maps
    single_sample = np.load("single_sample.npy", allow_pickle=True).item()

    # Access the individual images
    a_batch = single_sample['explanation']
    x_batch = single_sample['mfcc']
    y_batch = single_sample['ground_truth']

    print("Shape of x_batch originally:", x_batch.shape)
    print("Shape of a_batch originally:", a_batch.shape)
    print(y_batch)


    x_batch = np.expand_dims(np.expand_dims(x_batch, axis=0), axis=0)
    #x_batch = np.transpose(x_batch, (0, 3, 1, 2))
    a_batch = np.expand_dims(a_batch, axis=0)

    print("Shape of x_batch after:", x_batch.shape)
    print("Shape of a_batch after:", a_batch.shape)
    print(y_batch)

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

    faithfulness_scores = calculate_faithfulness(model, x_batch, y_batch, a_batch, device)
    print("Faithfulness Score:", faithfulness_scores)

    # Convert numpy arrays to torch tensors
    x_batch = torch.tensor(x_batch, dtype=torch.float32)
    a_batch = torch.tensor(a_batch, dtype=torch.float32)
    y_batch = torch.tensor(y_batch, dtype=torch.long)

    # Calculate performance impact
    impacts = calculate_performance_impact(model, x_batch[0], y_batch[0], a_batch[0], device)
    print("Performance Impacts:", impacts)

    # Plot the performance impact for one sample
    plot_performance_impact(impacts)



if __name__ == "__main__":
    main()