import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_model, dicom_to_tensor


"""
    This script takes in a user-specified path to a directory of A4C DICOMs. Predictions 
    are saved as a csv file in the directory that the script is run from. 
    The predictions file has 3 columns: 
        1. filename - name of the DICOM file 
        2. prediction - sigmoided output of the model
"""

NEURON_NAMES = ["none or trace", "mild", "moderate", "severe"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--weights_path", type=str, required=True)
args = parser.parse_args()

data_path = Path(args.data_path)
weights_path = Path(args.weights_path)
if not data_path.exists():
    print("File not found. Please enter path to an existing directory")
else:

    ### Create Pytorch dataset from user-inputted directory of DICOM files
    ### Convert each DICOM to tensor as input to model

    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, data_path: Path):
            self.data_path = data_path
            self.files = list(data_path.iterdir())

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            return self.files[idx].name, dicom_to_tensor(self.files[idx])

    dataset = InferenceDataset(data_path)

    print(f"Dataset length:\t{len(dataset)}")

    ### Set up dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        ### Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(device, weights_path=weights_path, num_classes=len(NEURON_NAMES))
    
        filenames = []
        predictions = []
        ### Run inference on each batch
        for idx, (batch_filenames, batch_tensor) in tqdm.tqdm(enumerate(dataloader)):
            batch_tensor = batch_tensor.to(device)
            output = torch.softmax(model(batch_tensor), dim=1)  # output from model
            predictions.append(output.cpu())  # output from model
            filenames.extend(batch_filenames)
    
        predictions = torch.cat(predictions, dim=0).T  # num_neurons x num_samples
        ### Create dataframe with columns for filename and predictions
        dataframe = {"filename": filenames}
    
        for key, value in zip(NEURON_NAMES, predictions):
            dataframe[key] = value
    
        dataframe = pd.DataFrame(dataframe)
        ### Save predictions to file
        dataframe.to_csv(f"predictions_{weights_path.stem}.csv", index=None)
