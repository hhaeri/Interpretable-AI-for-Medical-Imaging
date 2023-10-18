import torch
from pathlib import Path
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor
from ...CLAP-Git.CLAP-interpretable-predictions-main.src.architecture.clap import CLAP
from ...CLAP-Git.CLAP-interpretable-predictions-main.src.data.loading import get_datasets, resize_transform
from typing import Any, Callable, List, Optional, Union

import PIL
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset



RESULTS_DIR = Path("../CLAP-Git/resolution64")

#from .chestxray import ROOT_DIR
ROOT_DIR = Path("../ NIHCC_CRX_Full_Res / CXR8")

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


class RandomChestXRay(VisionDataset):
    def __init__(
        self,
        train: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_random_xrays = 1,
    ) -> None:
        super().__init__(str(ROOT_DIR), transforms, transform, target_transform)
        self.train = train

        self.filename, self.y = self._load_data()

    def _load_data(self) -> Union[List, np.ndarray]:
        idx_file = (
            ROOT_DIR / "train_val_list.txt"
            if self.train
            else ROOT_DIR / "test_list.txt"
        )
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

        return filename, y

    def __len__(self) -> int:
        return len(self.filename)

    def __getitem__(self, index: int) -> Any:
        y = self.y[index, ...]

        img_file = self.filename[index]
        x = PIL.Image.open(ROOT_DIR / "images" / "images" / img_file)
        
##HH chnaged transform to transforms here
        if self.transforms is not None:
            x = self.transforms(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y



random_dataset = RandomChestXRay(
    train=False,
    transforms=Compose(
            [
                # some images have 1 channel, others 4. Make them all 1 channel.
                Grayscale(num_output_channels=1),
                resize_transform(dataset_name, 64),
            ]
        ),
)
# Grabbed the dataset statistics from loading.py
setattr(random_dataset, "mean", [0.0])
setattr(random_dataset, "std", [1.0])

###### Load the Model

# Create an instance of the VAE model

n_channels, image_dim, n_classes = 1, 64, 15
z_style_dim, z_core_dim = 20, 10 

Clap_model = CLAP(n_channels, image_dim, z_style_dim, z_core_dim, n_classes)


# Load the model weights from the checkpoint file

checkpoint_path = RESULTS_DIR/"model_best.pth.tar.gz" 
checkpoint = torch.load(checkpoint_path)
Clap_model.load_state_dict(checkpoint['state_dict'])

###### Encode Data

# Expand dimensions if needed (e.g., for a single image)
if len(random_dataset.shape) == 2:
    random_dataset = random_dataset.unsqueeze(0)

# Pass the input through the encoder to obtain latent representations

mean_core, log_var_core, z_core, mean_style, log_var_style, z_style, x_reconstructed, y_pred = Clap_model.pred_vae(random_dataset)


fig, axs = plt.subplots(1,7)#, figsize=(14, 2))  # Create a row of subplots for each variation of z_style
all_reconstructed_images = []

# Traverse the Latent Space. Pass the latent space to the decoder to obtain the reconstructed images
#z_core:
for dc in range (z_core_dim):
    core_images = [] # Stores the reconstructed images for the current dc
    for j in range (-3,4):
        temp = z_core[dc].clone()
        z_core[dc] += j*torch.exp(log_var_core[dc])

        z = torch.cat([z_core, z_style], dim=-1)
        #the following line needs correction to concatanate the constructed images instead of overwriting
        reconstructed_images = Clap_model.decoder(z)
        
        # Append the reconstructed image to the current ds_images list
        core_images.append(reconstructed_images[0].detach().cpu().numpy()) 

        #Restore the original z_core[ds]
        z_core[dc] = temp
        
    # Append the row of images for the current ds to the all_reconstructed_images list
    all_reconstructed_images.append(core_images)
    
# Stack the rows of images to create a single tensor
final_image_core = torch.Tensor(all_reconstructed_images)

# Display the reconstructed images in the corresponding subplot
axs[0].imshow(final_image_core.permute(1, 0, 2, 3).reshape(final_image_core.size(1), -1))  # Assuming you have one image in the batch


all_reconstructed_images = []
#z_style:
for ds in range (z_style_dim):
    style_images = [] # Stores the reconstructed images for the current dc
    for j in range (-3,4):
        temp = z_style[ds].clone()
        z_style[ds] += j*torch.exp(log_var_style[ds])

        z = torch.cat([z_core, z_style], dim=-1)
        #the following line needs correction to concatanate the constructed images instead of overwriting
        reconstructed_images = Clap_model.decoder(z)
        
        # Append the reconstructed image to the current ds_images list
        style_images.append(reconstructed_images[0].detach().cpu().numpy()) 

        #Restore the original z_core[ds]
        z_style[ds] = temp
        
    # Append the row of images for the current ds to the all_reconstructed_images list
    all_reconstructed_images.append(style_images)
    
# Stack the rows of images to create a single tensor
final_image_style = torch.Tensor(all_reconstructed_images)

# Display the reconstructed images in the corresponding subplot
axs[1].imshow(final_image_style.permute(1, 0, 2, 3).reshape(final_image_stylex.size(1), -1))  # Assuming you have one image in the batch
plt.axis('off')
plt.show()
    

