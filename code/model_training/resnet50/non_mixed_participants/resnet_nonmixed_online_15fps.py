import os
import time
import datetime
import traceback
import torch
import torchvision
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.optim import lr_scheduler
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split


# >>> ==================== Non Mixed Participant Changes ===================
## Library dependecies
import random

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader
import glob

files_to_ignore = [".DS_Store"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
plt.ion()   # interactive mode

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
# classes = ["Control", "Failure"] - not required 

# >>> ==================== Non Mixed Participant Changes ===================
class_labels = ["0", "1"]
# >>> ==================== Non Mixed Participant Changes ===================

# Set a random seed for CPU
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Set a random seed for CUDA (if available)
if train_on_gpu:
    torch.cuda.manual_seed(seed)


# >>> ==================== Non Mixed Participant Changes ===================
# A CustomDataset Class - to read in participants without having overlapping frames in a given fold's splits.

## Defining a Custom Dataset class - to read in the frames of participants belonging to a particular fold's split
## This is to make sure that during training/validation/testing - the frames of participants remain present in 1 particular split
## i.e: non-mixed participants
class CustomDataset(Dataset):
    ## Initialise the CustomDataset class object to have the labels and the paths to all the image frames
    ## of all the participants beloning to a given fold's train/val/test split.
    def __init__(self, participants, study_data_path, transform=None):
        self.participants = participants
        self.study_data_path = study_data_path
        self.transform = transform

        self.images = []
        self.image_paths = []
        self.labels = []

        for participant in self.participants:
            participant_path = f"{study_data_path}/{participant}/frames/"
            for class_label in class_labels:
                class_path = os.path.join(participant_path, class_label)
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    
                    ## Append only the image paths of all the frames for the given fold split's participant
                    ## You will read in the image only in batches using this initialized frame paths
                    self.image_paths.append(image_path)
                    
                    ## Correspondingly store the class the frame belongs to
                    label = int(class_label)
                    self.labels.append(label)
        self.n_samples = len(self.labels)

    def __len__(self):
        return self.n_samples
    
    ## When reading in the data during epoch steps - load the frames in batches - based on the given indexes.
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
## <<< ==================== Non Mixed Participant Changes ===================

## >>> ==================== Non Mixed Participant Changes ===================
## Creating participant splits based on custom split sizes based on the number of folds required.
def createDataSplits(participants, output_path, train_fold_size, val_fold_size, test_fold_size, seed_value=42):
    """ accepts participants of a given study, along with the participant split sizes

    Args:
        participants (arr): an array of participants of a given study
        train_fold_size (int): train fold splitting size
        val_fold_size (int): validation fold splitting size
        test_fold_size (int): test fold splitting size
        seed_value (int, optional): _description_. Defaults to 42.

    Returns:
        train_fold (arr): training participants in each fold
        val_fold (arr): validation participants in each fold
        test_fold (arr): test participants in each fold
    """
    try:
        ## Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)

        number_of_participants = len(participants)

        ## Shuffle the list of participants
        np.random.shuffle(participants)

        ## Initialize lists to store the train, and test participants for each fold
        train_folds = []
        val_folds = []
        test_folds = []

        ## Define the number of folds to be created
        number_of_folds = 5

        for i in range(number_of_folds):
            start_test_index = i * test_fold_size
            end_test_index = start_test_index + np.min([test_fold_size, len(participants) - start_test_index])

            test_participants = participants[start_test_index: end_test_index]

            ## Identify all the participants except the participants belonging to the test_participants and shuffle them
            remaining_participants = np.setdiff1d(participants, test_participants)
            np.random.shuffle(remaining_participants)

            ## Validation set selected from the remaining participants
            validation_participants = remaining_participants[: val_fold_size]

            ## Identify all the participants except the participants belonging to the test fold
            train_participants = np.setdiff1d(remaining_participants, validation_participants)

            ## Append the participants sets to their corresponding folds
            train_folds.append(train_participants)
            val_folds.append(validation_participants)
            test_folds.append(test_participants)

        # # Sanity checks
        # print(f"Participants are : {participants}")
        # print(f"Training Fold Participants: {train_folds}")
        # print(f"Validation Fold Participants: {val_folds}")
        # print(f"Testing Fold Participants: {test_folds}")

        ## For now, consider just normal split (i.e: 1-fold)
        # train_fold = train_folds[0]
        # val_fold = val_folds[0]
        # test_fold = test_folds[0]

        return train_folds, val_folds, test_folds
    except Exception as e:
        with open(f"{output_path}/results_output_log.txt", "a") as results_log_file:
            results_log_file.write(
                f"Exception {e} thrown during splitting dataset for:- \n"
                f"{traceback.format_exc()}"
            )

## <<< ==================== Non Mixed Participant Changes ===================


def plot_batch_size_accuracy(train_accuracy, val_accuracy, batch_size, axs):
    """creates subplots of training & validation accuracy scores for varying batch sizes

    Args:
        train_accuracy (arr): training accuracies of the given model
        val_accuracy (arr): validation accuracies of the given model
        batch_size (arr): the batch sizes used
    """
    epochs = range(1, len(train_accuracy) + 1)

    # Convert tensors to NumPy arrays after moving them to CPU
    train_accuracy = [acc.cpu().numpy() for acc in train_accuracy]
    val_accuracy = [acc.cpu().numpy() for acc in val_accuracy]

    axs.plot(epochs, train_accuracy, label='Training Accuracy')
    axs.plot(epochs, val_accuracy, label='Validation Accuracy')
    axs.set_title(f'Batch Size: {batch_size}')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Accuracy')
    axs.legend()



def test_model(model, test_loader, criterion, model_output):
    print("Entered model testing")
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    #MODEL SAVE
    classes = ["Control", "Failure"]

    ## To store all the labels for generating classification report
    true_labels = []
    pred_labels = []
    
    # Set the model to evaluation mode
    model.eval()

    
    # iterate over test data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs)

        # calculate the batch loss
        loss = criterion(output, labels)

        # update test loss 
        test_loss += loss.item()*inputs.size(0)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(correct_tensor.numpy())

        # append true and predicted labels for the batch for classification report
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(pred.cpu().numpy())

        # calculate test accuracy for each object class
        for i in range(len(labels.data)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Completed model testing")
    print(f"Test Loss: {test_loss:.6f}\n")

    for i in range(2):
        if class_total[i] > 0:
            test_accuracy = (class_correct[i] / class_total[i]) * 100
            correct_pred = np.sum(class_correct[i])
            class_total_sum = np.sum(class_total[i])
            print(f"Test Accuracy of {classes[i]}: {test_accuracy} ({correct_pred}/{class_total_sum})")
        else:
            print(f"Test Accuracy of {classes[i]}: N/A (no training examples)' % ({classes[i]})")

    overall_test_accuracy = (np.sum(class_correct) / np.sum(class_total)) * 100
    overall_correct_preds = np.sum(class_correct)
    overall_class_total_sum = np.sum(class_total)
    print(f'\nTest Accuracy (Overall): {overall_test_accuracy} ({overall_correct_preds}/{overall_class_total_sum})')

    return true_labels, pred_labels, (overall_test_accuracy / 100), test_loss



def train_model(study_a_participants, study_a_data_path, study_b_data_path, study_a_splits,
                epoch_list, batch_sizes, model_output, model_last_save, resume=0, resume_fold=0, new_fold=0):
    
    print("Entered model training")

    ## >>> ==================== Non Mixed Participant Changes ===================
    ### Given the fold's split sizes - get the participants beloning to the given splits
    train_folds, val_folds, test_folds = createDataSplits(
        participants=study_a_participants,
        output_path=model_output, 
        train_fold_size=study_a_splits["train_fold_size"], 
        val_fold_size=study_a_splits["val_fold_size"], 
        test_fold_size=study_a_splits["test_fold_size"], 
        seed_value=42
        )
    print(f"Created model splits")

    ### Sanity Checks
    print("--" * 20)

    print(f"# of Train Folds : {len(train_folds)}")
    print(f"# of Val Folds : {len(val_folds)}")
    print(f"# of Test Folds : {len(test_folds)}")

    print("--" * 20)
    print(f"Train First Fold Participants: {train_folds[0]}")
    print(f"Val First Fold Participants: {val_folds[0]}")
    print(f"Test First Fold Participants: {test_folds[0]}")

    print("--" * 20)
    print(f"# of Train Fold Participants in First Fold: {len(train_folds[0])}")
    print(f"# of Val Fold Participants in First Fold: {len(val_folds[0])}")
    print(f"# of Test Fold Participants in First Fold: {len(test_folds[0])}")

    ### Define the transformations to be applied to the fold splits
    ## Define data transformations for training
    ## Perform Data Augmentation + Normalization
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ## Define data transformations for validation
    ## Perform Data Normalization
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ## Define data transformations for testing
    ## Perform Data Normalization
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    for fold in range(len(train_folds)):
        print("--" * 20)
        print(f"Fold : {fold + 1} / {len(train_folds)}")
        print("--" * 20)

        if fold != new_fold:
            print(f"Skipping fold {fold} as it is not the fold to start from")
            continue



       
        classes = ["Control", "Failure"]
        ## Define the model parameters
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2) ## replace the number of output nodes in the last fully connected layer.
        model = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        #CHANGE - changed optimized based on wandb sweep
        #optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
        optimizer = optim.Adadelta(model_ft.parameters(), lr=0.001)

        # Decay LR by a factor of 0.1 every 10 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # print(model_ft)

        ## Create the train/val/test split dataset for the given fold
        train_dataset = CustomDataset(participants=train_folds[fold], study_data_path=study_a_data_path, transform=transform_train)
        val_dataset = CustomDataset(participants=val_folds[fold], study_data_path=study_a_data_path, transform=transform_val)
        test_dataset = CustomDataset(participants=test_folds[fold], study_data_path=study_a_data_path, transform=transform_test)

        image_datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

        print(f"# of frames in training dataset {len(train_dataset)}")
        print(f"# of frames in validation dataset {len(val_dataset)}")
        print(f"# of frames in testing dataset {len(test_dataset)}")

        with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
            results_log_file.write(f"# of frames in training dataset {len(train_dataset)}\n")
            results_log_file.write(f"# of frames in validation dataset {len(val_dataset)}\n")
            results_log_file.write(f"# of frames in testing dataset {len(test_dataset)}\n")

        since = time.time()
        ## Count variable to keep track of the number of combinations in the hyper-parameter tuning
        search_count = 0
        
        # Create a figure with subplots for plotting the training and validation accuracy for various batch_sizes against the # of epochs
        num_cols = 2
        num_rows = 2
        batch_figure, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))

        #MODEL SAVE
        if resume == 1:
            if resume_fold == new_fold:
                #if resume_fold is greater than fold, then skip
                if resume_fold > fold:
                    print(f"Skipping fold {fold} as it is not the fold to resume from")
                    continue

                
                checkpoint = torch.load(f"{model_last_save}model_variables_{resume_fold}.pth")
                epoch_restart = checkpoint['epoch']
                train_losses = checkpoint['train_losses']
                train_accuracies = checkpoint['train_accuracies']
                val_losses = checkpoint['val_losses']
                val_accuracies = checkpoint['val_accuracies']
                model.load_state_dict(torch.load(f"{model_last_save}model_{resume_fold}_{epoch_restart}.pth"))
                print(f"Resuming from epoch {epoch_restart}")
                print(f"Resuming from fold {fold}")
                resume = 0
            else:
                epoch_restart = 1
                train_losses = []
                train_accuracies = []
                val_losses = []
                val_accuracies = []
        else:
                epoch_restart = 1
                train_losses = []
                train_accuracies = []
                val_losses = []
                val_accuracies = []


        print(f"Epochs List: {epoch_list}")
        for epochs in epoch_list:
            print(f"At beginning: epoch = {epochs}")
            #MODEL SAVE
            #if epochs == 0, then skip it
            if epochs == 0:
                continue
            if epoch_restart >= epochs:
                continue
            for i, batch_size in enumerate(batch_sizes):
                if batch_size == 0:
                    continue

                try:
                    search_count += 1
                    
                    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
                    ## To keep track of the accuracies to obtain the accuracy plots
                    #train_accuracies = []
                    #val_accuracies = []
                    #train_losses = []
                    #val_losses = []

                    dataloaders = {
                        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True) for x in ['train', 'val', 'test']
                    }
                    
                    # Create a temporary directory to save training checkpoints
                    tempdir = model_output  
                        ###CHANGE deleted savings
                    best_accuracy = 0.0

                    
                    print(f"Epoch_restart = {epoch_restart}")
                    #MODEL SAVE - EPOCH RESTART
                    for epoch in range(epoch_restart, epochs + 1):
                        print(f"Epoch {epoch}/{epochs}")
                        print("-" * 10)

                        ## Each epoch has a training and validation phase
                        for phase in ["train", "val"]:
                            print(f"Phase: {phase}")
                            if phase == "train":
                                model.train() # Set model to training mode
                            else:
                                model.eval() # Set model to evaluate mode
                        
                            running_loss = 0.0
                            running_corrects = 0
                            n_batch_iterations = len(dataloaders[phase])

                            ## Iterate over the data
                            batch_count = 1
                            for inputs, labels in dataloaders[phase]:
                                inputs = inputs.to(device)
                                labels = labels.to(device)

                                # zero the parameter gradients
                                optimizer.zero_grad()

                                # forward
                                # track history if only in train
                                with torch.set_grad_enabled(phase == 'train'):
                                    outputs = model(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                    # backward + optimize only if in training phase
                                    if phase == 'train':
                                        loss.backward()
                                        optimizer.step()

                                # statistics
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)
                                
                                if batch_count % 5 == 0:
                                    print(f"Epoch: {epoch}/{epochs}; Step: {batch_count}/{n_batch_iterations}")
                                batch_count += 1
                                # break ## break from iterating over the batches

                            if phase == "train":
                                scheduler.step()

                            epoch_loss = running_loss / dataset_sizes[phase]
                            epoch_acc = running_corrects.double() / dataset_sizes[phase]

                            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                            # CHANGE - commented below
                            # deep copy the model
                            #if phase == 'val' and epoch_acc > best_accuracy:
                            #    best_accuracy = epoch_acc
                            #    torch.save(model.state_dict(), best_model_params_path)

                            # Store accuracy for both training and validation
                            if phase == 'train':
                                train_accuracies.append(epoch_acc)
                                train_losses.append(epoch_loss)
                                
                            elif phase == 'val':
                                val_accuracies.append(epoch_acc)
                                val_losses.append(epoch_loss)
                                
                            # break ## break from iterating over model training and validation 
                            
                            #CHANGE - added model saving
                            #save model every 2 epochs after validating
                            if phase == 'val' and epoch % 2 == 0:
                                model_chpt = os.path.join(tempdir, f'model_chpt_{fold}_{epoch}.pth')
                                torch.save(model.state_dict(), model_chpt)
                                checkpoint = {
                                    'epoch': epoch,
                                    'train_losses': train_losses,
                                    'train_accuracies': train_accuracies,
                                    'val_losses': val_losses,
                                    'val_accuracies': val_accuracies,
                                    'classes': classes
                                }
                                torch.save(checkpoint, os.path.join(tempdir,f'model_variables_{fold}_{epoch}.pth'))
                                #MODEL SAVE
                                #save last model in last_model_save_dir
                                torch.save(model.state_dict(), f"{model_last_save}model_{fold}_{epoch}.pth")
                                torch.save(checkpoint, f"{model_last_save}model_variables_{fold}.pth")
                                #if it exists, delete the previous model
                                if os.path.exists(f"{model_last_save}model_{fold}_{epoch-2}.pth"):
                                    os.remove(f"{model_last_save}model_{fold}_{epoch-2}.pth")
                                

                        print()
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Acc: {best_accuracy:4f}')

                    #CHANGE - changed modelsaving
                    # load best model weights
                    # model.load_state_dict(torch.load(model_chpt))

                    print(f"Begin testing the model")
                    true_labels, pred_labels, test_accuracy, test_loss = test_model(model, dataloaders['test'], criterion, model_output)
                    print(f"Completed model testing")

                    report = classification_report(true_labels, pred_labels, target_names=classes)
                    print(report)
                    
                    # Calculate the confusion matrix
                    conf_matrix = confusion_matrix(true_labels, pred_labels)

                    # Plot the confusion matrix using seaborn and matplotlib
                    plt.figure(figsize=(8, 8))
                    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.title("Confusion Matrix")
                    
                    confusion_matrix_path = model_output + f"confusion_matrices/"

                    if not os.path.exists(confusion_matrix_path):
                        os.mkdir(confusion_matrix_path)
                    
                    confusion_matrix_path = confusion_matrix_path + f"confusion_matrix_{epoch}_{batch_size}.png"
                    plt.savefig(confusion_matrix_path)
                    plt.clf()

                    # ## Plot training and validation accuracy plots for varying batch_sizes
                    # ## Plot training and validation accuracy on the current subplot
                    # Calculate row and column indices for the current subplot
                    row_idx = i // num_cols
                    col_idx = i % num_cols

                    # Get the current subplot
                    ax = axs[row_idx, col_idx]
                    ax.clear() # clear the legends from previous plots
                    plot_batch_size_accuracy(train_accuracies, val_accuracies, batch_size, ax)

                    """
                    BEGIN: Subplots for training and validation accuracy for varying batch_sizes
                    """
                    # Adjust layout and save the figure for training & validation accuracy for varying batch_sizes
                    batch_size_results_path = model_output + 'batch_size_results/'

                    if not os.path.exists(batch_size_results_path):
                        os.makedirs(batch_size_results_path)

                    batch_size_accuracy_path = batch_size_results_path + f'batch_size_comparison_{epoch}.png'
                    batch_figure.tight_layout(pad=2.5)
                    batch_figure.savefig(batch_size_accuracy_path)
                    # plt.show()
                    plt.close(batch_figure)
                    plt.clf()
                    ###CHANGE - indentation change ends here

                    with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                        results_log_file.write("\n")
                        results_log_file.write(f"------------ lab 15FPS: BEGIN SEARCH: {search_count} ------------" + "\n")
                        results_log_file.write("------------ TYPE ------------" + "\n")

                        results_log_file.write(
                            f"Epoch: {epoch}\n"
                            f"Batch Size: {batch_size}\n"
                        )

                        results_log_file.write("------------ METRICS ------------" + "\n")
                        results_log_file.write(f'Training Loss: {train_losses[-1]:.4f}' + '\n')
                        results_log_file.write(f'Training Accuracy: {train_accuracies[-1]:.4f}' + '\n')
                        results_log_file.write(f'Validation Loss: {val_losses[-1]:.4f}' + '\n')
                        results_log_file.write(f'Validation Accuracy: {val_accuracies[-1]: .4f}' + '\n')
                        results_log_file.write(f'Test Loss: {test_loss:.4f}' + '\n')
                        results_log_file.write(f'Test Accuracy: {test_accuracy:.4f}' + '\n')

                        results_log_file.write("------------ CLASSIFICATION REPORT ------------" + "\n")
                        results_log_file.write(report + '\n')

                        results_log_file.write("------------ CONFUSION MATRIX ------------" + "\n")
                        results_log_file.write(f'Confusion matrix saved at {confusion_matrix_path}' + '\n')

                        results_log_file.write(f"------------ END SEARCH ------------" + "\n")
                        results_log_file.write("\n")

        

                except Exception as e:
                    with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                        results_log_file.write(
                            f"Search Count: {search_count} \n"
                            f"Epoch: {epoch}\n"
                            f"Batch Size: {batch_size}\n"
                            f"Exception {e} thrown during model training :- \n"
                            f"{traceback.format_exc()}\n"
                        )


                # break ## break from batch size
            # break ## break from epochs
            #MODEL SAVE - comment break
        #break ## break from folds
    return model

def main():
    

        color_channel = "BGR"
        data_frame_rate = 15
        dataset_path = "../../../../../data/study_data/"
        output_directory = "../../../../../data/"

        ## >>> ==================== Non Mixed Participant Changes ===================

        lab_participant_data_path = dataset_path + f"/non_mixed_lab_custom_dataset/data_prefix_{data_frame_rate}_fps/"
        lab_participants = [participant for participant in os.listdir(lab_participant_data_path) if participant not in files_to_ignore]
        
        ## Define the path for storing model outputs
        now = datetime.datetime.now()
        lab_output_path = output_directory + f"non_mixed_participants/lab/resnet50_lab_{data_frame_rate}_fps_" + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

        if not os.path.exists(lab_output_path):
            os.makedirs(lab_output_path)

        # 70/20/10 split - total 30 participants
        lab_fold_splits = {
            "train_fold_size": 21,
            "val_fold_size": 6,
            "test_fold_size": 3
        }

        ## >>> ==================== Non Mixed Participant Changes ===================

        #MODEL SAVE
        last_model_save_dir = output_directory + "non_mixed_participants/lab/" + f"resnet50_lab_nonmixed_{data_frame_rate}_fps_last_model/"

        
        
        new_fold = 0
        
        ############################################
        if not os.path.exists(last_model_save_dir):
            os.makedirs(last_model_save_dir)
            resume = 0
            resume_fold = 0
            
        else:
            #look at the models saved - they are in the format model_FOLD_EPOCH.pth. If epoch is less than 200, save variable "RESUME" as True. Look for each fold, start from 4 to 0
            for fold in range(4, -1, -1):
                if fold == new_fold:
                    #see if model exists, regardless of epoch (any epoch numbe is fine, look for partial string match with fold number and .pth extension)
                    path_start = f"{last_model_save_dir}model_{fold}_*.pth"
                    if glob.glob(path_start):
                        print('MODEL EXISTS FOR FOLD', fold)
                        
                        resume = 1
                        resume_fold = fold
                        
                        break
                    else:
                        resume = 0
                        resume_fold = 0
                        
        print("started from", resume_fold)
        print("resume is", resume)


        epoch_list = [30,50,70]
        batch_sizes = [32,64]

        model_ft = train_model(
            study_a_participants=lab_participants,
            study_a_data_path=lab_participant_data_path,
            study_b_data_path=lab_participant_data_path, ## Not really used currently - also, to be changed to the other study's data path - after creating the dataset.
            study_a_splits=lab_fold_splits,
            epoch_list=epoch_list,
            batch_sizes=batch_sizes,
            model_output=lab_output_path,
            model_last_save=last_model_save_dir,
            resume=resume,
            resume_fold=resume_fold,
            new_fold=new_fold
        )


if __name__ == "__main__":
    
    main()
