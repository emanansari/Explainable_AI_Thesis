import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
#from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from dataset import UrbanSound8K
from resnet101 import ResNetAudio101

#NEW FINAL - PROCESSES ALL 50 IMAGES IN ONE GO AND PRODUCES VISUALIZATION FOR 1

DATAPATH = '/home3/s4317394/pytorch/audio'
FOLDS_TEST = [10]
BATCH_SIZE = 50  # Process one sample at a time for better visualization
target_sr = 44100
n_mfcc = 13
max_length = 400

# Set torchaudio backend to soundfile
torchaudio.set_audio_backend("soundfile")

# Initialize the model
model = ResNetAudio101(num_classes=10)
model.load_state_dict(torch.load("cnn_model101.pth"))
model.eval()
print("ResNet model loaded and set to evaluation mode")

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using device {device}')

# Prepare the test dataset and dataloader
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=target_sr,
    n_mfcc=n_mfcc,
    melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 100},
)

test_dataset = UrbanSound8K(DATAPATH, device, folders_in=FOLDS_TEST, transform=mfcc_transform)

# Randomly select 50 samples from the test dataset
indices = random.sample(range(len(test_dataset)), 50)
test_subset = Subset(test_dataset, indices)
test_dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize GradCAM
target_layer = model.resnet.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Define class names
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
               'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Initialize lists to hold batch data
all_mfcc = []
all_ground_truth = []
all_explanations = []


# Process each sample in the test subset
for i, (inputs, labels) in enumerate(test_dataloader):
    input_tensor = inputs.to(device)
    
    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Convert predicted labels to CPU and extract as numpy array
    predicted = predicted.cpu().numpy()

    # Use the predicted label for GradCAM
    target_labels = predicted.tolist()  # Convert to list for indexing later

    for j, target_label in enumerate(target_labels):
        grayscale_cam = cam(input_tensor=input_tensor[j].unsqueeze(0), targets=[ClassifierOutputTarget(target_label)])

        # Process and visualize the result
        input_image = inputs[j].cpu().numpy()  # Squeeze batch dimension
        input_image = input_image[0]  # Assuming the MFCC spectrogram is the first channel

        # Normalize input image
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

        # Resize grayscale_cam to match input_image shape
        heatmap = grayscale_cam[0]
        
        # Ensure heatmap and input_image have compatible shapes
        if heatmap.shape != input_image.shape:
            print(f"Warning: Shapes mismatch! Heatmap shape: {heatmap.shape}, Input image shape: {input_image.shape}")
            continue  # Skip processing if shapes do not match
        
        heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        heatmap_save = heatmap.copy()

        # Append data to lists
        all_mfcc.append(input_image)
        all_ground_truth.append(labels[j].item())  # Get ground truth label for current item
        all_explanations.append(heatmap_save)


        # Plotting for the first sample in the batch only
        if j == 0:
            fig = plt.figure(figsize=(15, 10))  # Adjust figure size as needed
            
            # Colorbar
            ax_colorbar = fig.add_axes([0.1, 0.04, 0.8, 0.02])  # Position for colorbar
            norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
            sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='horizontal')
            cbar.set_label('Explanation Magnitude', fontsize=19)
            cbar.ax.tick_params(labelsize=19)

            # Heatmap image
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.imshow(heatmap, cmap='jet', aspect='auto')
            ax1.set_title('Heatmap', fontsize=19)
            ax1.axis('off')

            # Original MFCC image
            ax2 = fig.add_subplot(3, 1, 2)
            ax2.imshow(input_image, cmap='gray', aspect='auto')
            ax2.set_title(f'Original MFCC of Class: {class_names[labels[j].item()]}', fontsize=19)
            ax2.axis('off')

            # Overlayed image
            overlay_image = heatmap * 0.3 + input_image * 0.5
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.imshow(overlay_image, cmap='jet', aspect='auto')
            ax3.set_title(f'Overlayed Heatmap on MFCC, Predicted class: {class_names[target_label]}', fontsize=19)
            ax3.axis('off')

            # Adjust spacing between subplots
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjusted to ensure suptitle is visible

            # Print the plot title with the ground truth class name and predicted class name
            plot_title = f'Ground Truth: {class_names[labels[j].item()]}, Predicted: {class_names[target_label]}'
            print(f'Plot Title for Image {i * BATCH_SIZE + j + 1}: {plot_title}')
        
            # Save or display the figure
            plt.savefig(f"gradcam_explanations_101{i * BATCH_SIZE + j + 1}.png", bbox_inches='tight')
            plt.show()

# Convert lists to NumPy arrays
all_mfcc = np.array(all_mfcc)
all_ground_truth = np.array(all_ground_truth)
all_explanations = np.array(all_explanations)

# Convert lists to numpy arrays
all_mfcc = np.stack(all_mfcc, axis=0)
all_explanations = np.stack(all_explanations, axis=0)

# Print shapes
print("Shape of all_mfcc:", all_mfcc.shape)
print("Shape of all_ground_truth:", all_ground_truth.shape)
print("Shape of all_explanations:", all_explanations.shape)

# Save all the collected data in a single NumPy file
explanations_dict = {
    'mfcc': np.array(all_mfcc),
    'ground_truth': np.array(all_ground_truth),
    'explanation': np.array(all_explanations)
}
np.save("Explanations101_batch.npy", explanations_dict)
