import torch
from torch.utils.data import Dataset
import kagglehub
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Ready to train on: {device}")

# Downloading the data from Kaggle server to lightning ai server
path = kagglehub.dataset_download("nih-chest-xrays/data")

dataset_main_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
csv_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv'
test_list_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/test_list.txt'




class ChestXRaysDataset(Dataset):
    def __init__(self, data_df, dataset_path, transform = None):
        self.df = data_df
        self.transform = transform

        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        self.image_path = {}
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.png'):
                    self.image_path[file] = os.path.join(root, file)

        print(f"Dataset is initialized Correctly with length {len(self.df)}")
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['Image Index']
        image_path = self.image_path[image_name]   
        image = Image.open(image_path).convert('RGB')
        self.labels = []
        labels_str = self.df.iloc[idx]['Finding Labels']
        labels_vector = torch.zeros(len(self.classes))
        for i, disease in enumerate(self.classes):
            if disease in labels_str:
                labels_vector[i] = 1.0

        if self.transform is not None:
            image = self.transform(image)

        return image, labels_vector
        



def split_data(csv_path, test_list_path):
    df_data = pd.read_csv(csv_path)

    unique_patients = df_data['Patient ID'].unique()

    train_patients, val_patients = train_test_split(unique_patients, test_size = .2, random_state = 42)

    train_df = df_data[df_data['Patient ID'].isin(train_patients)].reset_index(drop = True)
    val_df = df_data[df_data['Patient ID'].isin(val_patients)].reset_index(drop = True)

    with open(test_list_path, 'r') as f:
        test_images = f.read().splitlines()

    print(f"Found {len(test_images)} images to test")

    test_df = df_data[df_data['Image Index'].isin(test_images)].reset_index(drop = True)
    

    return train_df, val_df, test_df