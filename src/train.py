n_classes = 14
batch_size = 16
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import gc
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os

from data_utils import ChestXRaysDataset
from model import ChexNet

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Ready to train on: {device}")

def epoch_train(model,train_loader,loss_function, optimizer):
    model.train()
    total_loss = 0.0
    
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def epoch_val(model, val_loader, loss_function):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = loss_function(output, label)
            total_loss += loss.item()
    return total_loss / len(val_loader)
            

def train(n_classes, batch_size, model_name, train_df, val_df, lr, num_epochs):
    if model_name == 'ChexNet':
        model = ChexNet(n_classes).to(device)
        model = torch.nn.DataParallel(model).to(device)

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(degrees = 15),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
    ])

    dataset_train = ChestXRaysDataset(train_df, dataset_main_path, train_transforms)
    dataset_val = ChestXRaysDataset(val_df, dataset_main_path, val_transforms)

    dataloader_train = DataLoader(dataset_train, shuffle = True, batch_size = batch_size, num_workers=2)
    dataloader_val = DataLoader(dataset_val, shuffle = False, batch_size = batch_size, num_workers=2)

    optimizer = optim.AdamW(lr = lr, params = model.parameters())
    loss = nn.BCEWithLogitsLoss()

    schedular = ReduceLROnPlateau(factor = .1, optimizer = optimizer, patience = 2, mode = "min")
    

    CHECKPOINT_PATH = "chexnet_checkpoint.pth"
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        print('Found an existing model...')
        checkpoint = torch.load(CHECKPOINT_PATH, map_location = device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        schedular.load_state_dict(checkpoint['schedular_state'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print('Model is reloaded Succesfully!')
    else:
        print('No Checkpoints exists, Starting Training...')

    for epoch in range(start_epoch, num_epochs):
        train_loss = epoch_train(model,dataloader_train,loss, optimizer)
        val_loss = epoch_val(model, dataloader_val, loss)

        schedular.step(val_loss)

        checkpoint = {
            'epoch' : epoch,
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'schedular_state' : schedular.state_dict(),
            'best_val_loss' : best_val_loss 
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved!")
        print(f"[----] Epoch {epoch+1} Train={train_loss:.4f} Val={val_loss:.4f}")
    

        


    
   

    
