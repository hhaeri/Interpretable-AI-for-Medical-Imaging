import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from CLAP.src.architecture.clap import CLAP
from CLAP.src.data.loading import get_datasets, resize_transform
from typing import Any, Callable, List, Optional, Union

import matplotlib.pyplot as plt

import random
import PIL
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset



RESULTS_DIR = Path("./resolution64_GPU_2")

#from .chestxray import ROOT_DIR
ROOT_DIR = Path("./NIHCC_ChestXray/CXR8")

dataset_name = "ChestXRay"
###### Load the Data

LABELS = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Effusion": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Hernia": 7,
    "Infiltration": 8,
    "Mass": 9,
    "No Finding": 10,
    "Nodule": 11,
    "Pleural_Thickening": 12,
    "Pneumonia": 13,
    "Pneumothorax": 14,
}

num_random_xrays = 1

def RandomChestXRay():

        idx_file = (
            ROOT_DIR / "test_list.txt")
        with open(idx_file, "r") as file:
            images = set(map(lambda s: s.strip("\n"), random.sample(file.readlines(), num_random_xrays)))
        info_df = pd.read_csv(
            ROOT_DIR / "Data_Entry_2017_v2020.csv", index_col="Image Index"
        )

        # select only images selected randomly in the test split
        info_df = info_df[info_df.index.isin(images)]
        filename = list(info_df.index)

        # extract labels
        info_df["Finding Labels"] = info_df["Finding Labels"].map(
            lambda label: label.split("|")
        )

        y = np.zeros((len(filename), len(LABELS)), dtype=np.int8)
        for i, (image_file, (index, row)) in enumerate(
            zip(filename, info_df.iterrows())
        ):
            assert index == image_file
            for disease in row["Finding Labels"]:
                y[i, LABELS[disease]] = 1

        #y = y[index, ...]

        img_file = filename[0] #for num_random_xrays = 1  
        x = PIL.Image.open(ROOT_DIR / "images" / img_file)
        
        transforms=Compose(
            [
                # some images have 1 channel, others 4. Make them all 1 channel.
                Grayscale(num_output_channels=1),
                resize_transform(dataset_name, 64), #this includes the ToTensor transformation
            ])

        x = transforms(x)


        # Expand dimensions if needed (e.g., for a single image)
        if len(x.shape) <4:
            x = x.unsqueeze(0)

        return x, y, img_file

def custom_transforms(x):

    # Create a custom transformation for min-max scaling
    min_max_scale = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x)

    # Create a custom transformation for sharpness adjustment
    sharpness_factor = 8.0  # Adjust as needed
    sharpness_adjust = transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, sharpness_factor))

    # Create a custom transformation for gamma correction
    gamma_factor = 0.8  # Adjust as needed (lower values make grays darker)
    gamma_correct = transforms.Lambda(lambda x: torch.pow(x, gamma_factor))

    # Create a custom transformation for clamping values within [0, 1]
    clamp_values = transforms.Lambda(lambda x: torch.clamp(x, 0, 1))

    # Create a custom transformation for final normalization
    normalize = transforms.Normalize(mean=[0.0], std=[1.0])  # No-op for visualization (values are already in [0, 1])

    # Create a list of transformations in a Compose object
    transform = transforms.Compose([
        min_max_scale,  # Step 1: Min-max scaling (rescale values to [0, 1])
        sharpness_adjust,  # Step 2: Adjust sharpness (modify sharpness factor as needed)
        gamma_correct,  # Step 3: Apply gamma correction with a lower gamma factor to make grays darker
        clamp_values,  # Step 4: Clamp values within [0, 1]
        normalize,  # Step 5: No-op normalization (for visualization)
    ])

    # return the transformed tensor
    return transform(x)[0,:,:,:]



def traverse_latents(feature,var_span,var_mult):
    if feature =='core':
        var_, z_ = log_var_core.clone(), z_core.clone()
    else:
        var_, z_ = log_var_style.clone(), z_style.clone()

    num_latents = len(z_[0])
    #print(num_latents)
    fig, axs = plt.subplots(num_latents, var_span*2+1, figsize=(1*(var_span*2+1),1*num_latents),gridspec_kw = {'wspace':0, 'hspace':0}) #(width,height)

    #print(axs.shape)
    for i in range(num_latents):
        for j in range(-var_span,var_span+1):
            temp = z_.clone()
            z_[0][i] +=   var_mult *j * torch.exp(var_[0][i])
            z = torch.cat([z_ if feature=='core' else z_core, z_ if feature=='style' else z_style], dim=-1)
            recon_image = Clap_model.decoder(z)
            z_ = temp
            transformed_image = custom_transforms(recon_image)
            # Plot each image as we reconstruct rather than storing
            #print(transformed_image.shape)
            axs[i,j+var_span].imshow(transformed_image[0,:,:].detach().numpy(), cmap='gray', aspect='auto')
            #axs[i, j + var_span].set_aspect('equal')  # Set the aspect ratio to 'equal'
            axs[i, j].axis('off')
            #axs[i,j].set_title(f"Latent {i}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.tight_layout()
    plt.show()




def predict_top_labels(top_k,logits):

    # Number of top predictions you want: top_k

    # Calculate class probabilities using softmax
    probabilities = F.softmax(logits, dim=0)


    # Find the top-k class indices and their corresponding probabilities
    top_indices = torch.topk(probabilities, top_k)

    top_class_indices = top_indices.indices
    top_class_probabilities = top_indices.values
    # Reverse the LABELS dictionary to map indices to labels
    INDEX_TO_LABEL = {v: k for k, v in LABELS.items()}
    # Get the corresponding class labels or class names
    top_classes = [INDEX_TO_LABEL[idx.item()] for idx in top_class_indices]

    # Create a list of (class, probability) tuples
    top_predictions = [(class_name, probability.item()) for class_name, probability in zip(top_classes, top_class_probabilities)]

    print("Top Predictions:")
    for class_name, probability in top_predictions:
        print(f"{class_name}: Probability = {probability:.4f}")



def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert tensor of logit values to binary labels."""

    return torch.where(logits > 0, 1, 0)


def labels_to_conditions(label):

    conditions = [l for l, value in LABELS.items() if label[0][value] != 0]
    return conditions


###### Load the Data

random_dataset,label,img_name = RandomChestXRay()
#conditions = [l for l, value in LABELS.items() if label[0][value] != 0]
print('image_name:',img_name,'True Lables:',labels_to_conditions(label))


###### Load the Model

# Create an instance of the VAE model

n_channels, image_dim, n_classes = 1, 64, 15
z_style_dim, z_core_dim = 20, 10 

Clap_model = CLAP(n_channels, image_dim, z_style_dim, z_core_dim, n_classes)


# Load the model weights from the checkpoint file

checkpoint_path = RESULTS_DIR/"model_best.pth.tar.gz" 
checkpoint = torch.load(checkpoint_path)
Clap_model.load_state_dict(checkpoint['model_state_dict'])

###### Encode Data

# Pass the input through the encoder to obtain latent representations

mean_core, log_var_core, z_core, mean_style, log_var_style, z_style, x_reconstructed, y_pred = Clap_model.pred_vae(random_dataset).values()

# print the predicted labels
print("Predicted Labels:",labels_to_conditions(logits_to_labels(y_pred)))
predict_top_labels(3,y_pred[0])

# Traverse the latent space and visualize the reconstructed images
feature,var_span,var_mult = 'core', 3,10
traverse_latents(feature,var_span,var_mult)
