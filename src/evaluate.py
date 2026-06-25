
pathologies = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
                'Pleural Thickening', 'Hernia']

import torch.nn as nn
from model import ChexNet
from data_utils import ChestXRaysDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.transforms as transforms

dataset_main_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3'
csv_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv'
test_list_path = '/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/test_list.txt'
model_path = 'best_model.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Ready to train on: {device}")

def test_model(model_path, test_df, num_classes, batch_size):
    checkpoint = torch.load(model_path, map_location = device)
    model = ChexNet(num_classes).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.eval()

    prediction = []
    ground_truth = []

    test_trasforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
    ])
    
    dataset_test = ChestXRaysDataset(test_df, dataset_main_path, test_trasforms)
    dataloader_test = DataLoader(dataset_test, shuffle = True, batch_size = batch_size, num_workers=2)
    

    with torch.no_grad():
        for image, labels in dataloader_test:
            image = image.to(device)
            labels = labels.to(device)

            output = model(image)
            propa = torch.sigmoid(output)
            
            prediction.append(propa.cpu())
            ground_truth.append(labels.cpu())

    final_pred = torch.cat(prediction, dim =0)
    final_ground_truth = torch.cat(ground_truth, dim =0)

    return final_pred, final_ground_truth
            


def AUC(predictions, ground_truth, num_classes, classes_names):
    pred = predictions.numpy()
    gt = ground_truth.numpy()

    aucroc_scores = []

    for i in range(num_classes):
        p = pred[:,i]
        y = gt[:,i]
        disease_name = classes_names[i]
        scores = roc_auc_score(y, p)
        aucroc_scores.append(scores)
        print(f"AUC-ROC for {disease_name} : {scores}")

    avg_scores = np.mean(aucroc_scores)
    print(f"Average AUC-ROC is {avg_scores}")

    return avg_scores


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def create_curves():    
    plt.figure(figsize=(10, 8))

    if torch.is_tensor(final_ground_truth):
        final_ground_truth = final_ground_truth.cpu().numpy()
    if torch.is_tensor(final_pred):
        final_pred = final_pred.cpu().numpy()

    for i in range(14):
        fpr, tpr, thresholds = roc_curve(final_ground_truth[:, i], final_pred[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{pathologies[i]} (AUC = {roc_auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Guessing (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Multi-Class ROC Curves for CheXNet Validation', fontsize=16, fontweight='bold', pad=20)

    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('chexnet_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("ROC curves saved succesfully in : chexnet_roc_curves.png")




# Grad-CAM Class to extract the magic from your model
class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to grab the gradients and features during forward/backward passes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.eval()
        model_output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Get the score for the specific disease and backpropagate
        target = model_output[0][target_class]
        target.backward()

        # Pool the gradients and weight the activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # Create the heatmap, apply ReLU (only keep positive influences)
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        return heatmap

# apply the heatmap on the x-ray image
def show_cam_on_image(img_path, heatmap, save_path="gradcam_result.png"):
    
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    original_img = np.float32(original_img) / 255.0

    heatmap = cv2.resize(heatmap, (224, 224))
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255.0
    
    
    overlay = heatmap_colored * 0.4 + original_img * 0.6
    overlay = overlay / np.max(overlay) # Re-normalize
    
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img[:, :, ::-1])
    axes[0].set_title('Original X-Ray', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(overlay[:, :, ::-1])
    axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved successfully as: {save_path}")



# --- Setup ---

if __name__ == '__main__':

    checkpoint = torch.load(model_path, map_location = device)
    model = ChexNet(14).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint)

    target_layer = model.module.denseNet.features.denseblock4.denselayer16.conv2

    cam = GradCam(model=model, target_layer=target_layer)

    # --- Prepare the Image ---
    # Pick an image path from your test set that you know has a specific disease (e.g., Cardiomegaly)
    image_path = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3/images_001/images/00000013_010.png"

    # Same transforms you used in your dataset class
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_pil = Image.open(image_path).convert('RGB')
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # --- Generate Heatmap ---
    # Let's say you want to see where the model looks for 'Cardiomegaly'
    # Look up the index of Cardiomegaly in your pathologies list (e.g., index 1)
    disease_name = 'Pneumonia' 

    target_class_index = pathologies.index(disease_name)

    heatmap = cam.generate(input_tensor, target_class=target_class_index)

    # Overlay and Save!
    show_cam_on_image(image_path, heatmap, save_path="Cardiomegaly_gradcam.png")