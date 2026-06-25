import os
import kagglehub
from data_utils import split_data
from train import train
from evaluate import test_model, AUC, create_curves
from config import dataset_main_path, csv_path, test_list_path, model_path
def main():

    # dataset_main_path = kagglehub.dataset_download("nih-chest-xrays/data")
    
    # csv_path = os.path.join(dataset_main_path, 'Data_Entry_2017.csv')
    # test_list_path = os.path.join(dataset_main_path, 'test_list.txt')
    #=========

    if not os.path.exists(dataset_main_path):
        print("Data not found, downloading...")
        kagglehub.dataset_download("nih-chest-xrays/data")
    else:
        print(f"Data found at {dataset_main_path}, skipping download.")

    train_df, val_df, test_df = split_data(csv_path, test_list_path)

    pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    
    N_CLASSES = 14
    BATCH_SIZE = 32
    LR = 1e-3
    NUM_EPOCHS = 20

   
    MODE = 'test' 

    if MODE == 'train':
        print("--- Starting Training Phase ---")
        train(
            n_classes=N_CLASSES, 
            batch_size=BATCH_SIZE, 
            model_name='ChexNet', 
            train_df=train_df, 
            val_df=val_df, 
            lr=LR, 
            num_epochs=NUM_EPOCHS,
            dataset_main_path=dataset_main_path
        )
    
    elif MODE == 'test':
        print("--- Starting Evaluation Phase ---")        
        final_pred, final_ground_truth = test_model(model_path, test_df, N_CLASSES, BATCH_SIZE)
        f1 = final_pred
        f2 = final_ground_truth
        AUC(f1, f2, N_CLASSES, pathologies)
        
        create_curves(f1, f2)

if __name__ == '__main__':
    main()