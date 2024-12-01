import torch
import numpy as np
import quantus
from resnet import ResNetAudio  # Assuming ResNetAudio is defined in resnet.py
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from scipy.stats import spearmanr

# Initializing the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def perturb_input(x, a, baseline_type="zero", device="cpu"):
    perturbed_x = x.clone().to(device)
    a = a.to(device)
    
    if baseline_type == "zero":
        mask = torch.bernoulli(a).to(device)
        perturbed_x = perturbed_x * (1 - mask)
    
    return perturbed_x, mask

def normalize_attributions(a):
    a_min = a.min()
    a_max = a.max()
    normalized_a = (a - a_min) / (a_max - a_min)
    return normalized_a

def calculate_uncertainty(model, x, device):
    model.eval()
    with torch.no_grad():
        output = model(x.unsqueeze(0).to(device))
        probabilities = torch.softmax(output, dim=1)
        uncertainty = 1.0 / torch.max(probabilities)
    return uncertainty.item()

def calculate_faithfulness_single_sample(model, x_sample, a_sample, y_sample, device, nr_samples=10):
    model.eval()
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)
    original_output = model(x_sample.unsqueeze(0))
    original_uncertainty = 1.0 / torch.max(torch.softmax(original_output, dim=1))

    impacts = []
    perturbation_changes = []
    for step in range(nr_samples):
        perturbed_x, mask = perturb_input(x_sample, a_sample, baseline_type="zero", device=device)
        perturbed_output = model(perturbed_x.unsqueeze(0))
        perturbed_uncertainty = 1.0 / torch.max(torch.softmax(perturbed_output, dim=1))

        # Calculate impact (change in uncertainty)
        impact = original_uncertainty - perturbed_uncertainty
        impacts.append(impact.item())

        # Calculate change in perturbed attributions
        perturbed_attributions = a_sample * (1 - mask)
        perturbation_change = a_sample - perturbed_attributions
        perturbation_changes.append(perturbation_change.cpu().numpy())

    # Stack perturbation changes
    perturbation_changes = np.stack(perturbation_changes, axis=0)

    # Calculate Spearman's correlation between steps and impacts
    steps = range(len(impacts))
    correlation, _ = spearmanr(steps, impacts)
    print(f"Spearman's correlation: {correlation}")

    return impacts, perturbation_changes, correlation

def calculate_faithfulness(model, x_batch, y_batch, a_batch, device):
    faithfulness = quantus.MonotonicityCorrelation(
        nr_samples=10,
        features_in_step=13,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
    )(model=model, 
      x_batch=x_batch,
      y_batch=y_batch,
      a_batch=a_batch,
      device=device)
            
    return faithfulness

def save_single_sample(sample_data, sample_index):
    # Extract the individual samples
    a_batch = sample_data['explanation']
    x_batch = sample_data['mfcc']
    y_batch = sample_data['ground_truth']

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
    np.save(f"single_sample.npy", single_sample_data)
    print("Single sample saved as single_sample.npy")
    print(sample_index)

def main():
    # Load the pre-computed attribution maps
    sample_data = np.load("Explanations_batch.npy", allow_pickle=True).item()

    X_batch = sample_data['mfcc']
    sample_index = random.randint(0, X_batch.shape[0] - 1)

    # Save a single sample (e.g., the first sample)
    save_single_sample(sample_data, sample_index)

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
    a_batch = np.expand_dims(a_batch, axis=0)

    print("Shape of x_batch after:", x_batch.shape)
    print("Shape of a_batch after:", a_batch.shape)
    print(y_batch)

    print("Type of y_batch:", type(y_batch))

    if not isinstance(y_batch, np.ndarray):
        y_batch = np.array([y_batch])
    print("Type of y_batch after conversion:", type(y_batch))
    print("Shape of y_batch after conversion:", y_batch.shape)

    # Load your pre-trained model
    model = ResNetAudio(num_classes=10)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    model.eval()
    model = model.to(device)

    faithfulness_scores = calculate_faithfulness(model, x_batch, y_batch, a_batch, device)
    print("Faithfulness Scores:", faithfulness_scores)

    # Normalize the attributions
    a_batch = normalize_attributions(a_batch)

    # Convert numpy arrays to torch tensors
    x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
    a_batch = torch.tensor(a_batch, dtype=torch.float32).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    # Calculate faithfulness for the single sample
    impacts, perturbation_changes, spearman_corr = calculate_faithfulness_single_sample(model, x_batch[0], a_batch[0], y_batch[0], device=device)
    print("Impacts:", impacts)
    print("Spearman's correlation:", spearman_corr)

    # Flatten attribution map for plotting
    a_flattened = a_batch.cpu().numpy().flatten()

    # Calculate average absolute perturbation changes
    average_perturbation_changes = np.mean(np.abs(perturbation_changes), axis=0)

    # Plot change in perturbed attributions against impacts
    plt.figure(figsize=(8, 6))
    for i in range(len(perturbation_changes)):
        # Flatten and repeat impacts for plotting
        impacts_repeated = np.repeat(impacts[i], perturbation_changes.shape[1] * perturbation_changes.shape[2])

        # Flatten and repeat perturbation changes
        perturbation_changes_flattened = perturbation_changes[i].flatten()

        plt.scatter(perturbation_changes_flattened, impacts_repeated, alpha=0.5, label=f'Step {i+1}')
    print("Shape of perturbation_changes:", perturbation_changes_flattened.shape)

    # Plot the average change in perturbed attributions
    plt.scatter(average_perturbation_changes.flatten(), np.zeros_like(average_perturbation_changes.flatten()), color='r', marker='x', label='Average Change')
    print("Shape of perturbation_changes:", perturbation_changes_flattened.shape)

    plt.title('Impacts vs. Change in Perturbed Attributions')
    plt.xlabel('Change in Perturbed Attributions')
    plt.ylabel('Impacts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("impacts_vs_perturbed_attributions.png")
    plt.show()

if __name__ == "__main__":
    main()
