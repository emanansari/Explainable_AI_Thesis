import torch
import torchaudio
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from dataset import UrbanSound8K
from resnet import ResNetAudio

#currently the final version of gradcam XAI, generates both plots and npy files

DATAPATH = '/home3/s4317394/pytorch/audio'
FOLDS_TEST = [10]
BATCH_SIZE = 1  # Process one sample at a time for better visualization
target_sr = 44100
n_mfcc = 13
max_length = 400

# Set torchaudio backend to soundfile
torchaudio.set_audio_backend("soundfile")

# Initialize the model
model = ResNetAudio(num_classes=10)
model.load_state_dict(torch.load("cnn_model.pth"))
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

# Randomly select 5 samples from the test dataset
indices = random.sample(range(len(test_dataset)), 5)
test_subset = Subset(test_dataset, indices)
test_dataloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize GradCAM
target_layer = model.resnet.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Define class names
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 
               'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']



# Process each sample in the test subset
for i, (inputs, labels) in enumerate(test_dataloader):
    input_tensor = inputs.to(device)
    
    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Use the predicted label for GradCAM
    target_label = predicted.item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_label)])

    # Process and visualize the result
    input_image = inputs.squeeze(0).cpu().numpy()  # Squeeze batch dimension
    input_image = input_image[0]  # Assuming the MFCC spectrogram is the first channel

    # Normalize input image
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    # Resize grayscale_cam to match input_image shape
    heatmap = grayscale_cam[0]
    heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
    heatmap_save = heatmap.copy()

    # Ensure heatmap and input_image have compatible shapes
    if input_image.ndim == 2:
        input_image = np.expand_dims(input_image, axis=-1)  # Expand dimensions for compatibility
    #if input_image.shape[-1] != 3:  # Ensure input_image is RGB or single-channel (grayscale)
    #   input_image = np.tile(input_image, (1, 1, 3))  # Convert to RGB if single-channel

    # Apply colormap to heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on the input image
    cam_result = heatmap * 0.3 + input_image * 0.5
    cam_result = np.uint8(255 * cam_result)

    # Save the data into a dictionary
    explanations = {
        'mfcc': input_image,
        'ground_truth': labels.item(),
        'heatmap': heatmap,
        'explanation': heatmap_save
    }

    # Save the dictionary as a NumPy file
    np.save(f"Explanations{i+1}.npy", explanations)


    # Plotting all three images in one figure vertically
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

    # Original MFCC image
    axs[0].imshow(input_image, cmap='gray')
    axs[0].set_title(f'Original MFCC of Class: {class_names[labels.item()]}')
    axs[0].axis('off')

    # Heatmap image
    im = axs[1].imshow(heatmap, cmap='jet')
    axs[1].set_title('Heatmap')
    axs[1].axis('off')

    # Add colorbar to the heatmap
    cbar = plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label('Explanation Magnitude')

    # Overlayed image
    axs[2].imshow(cam_result)
    axs[2].set_title(f'Overlayed Heatmap on MFCC, Predicted class: {class_names[target_label]}')
    axs[2].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusted to ensure suptitle is visible
    
    # Print the plot title with the ground truth class name and predicted class name
    plot_title = f'Ground Truth: {class_names[labels.item()]}, Predicted: {class_names[target_label]}'
    print(f'Plot Title for Image {i+1}: {plot_title}')
   
    # Save or display the figure
    plt.savefig(f"gradcam_explainations_{i+1}.png")
    plt.show()
