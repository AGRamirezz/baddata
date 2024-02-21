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



import glob


files_to_ignore = [".DS_Store"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
plt.ion()   # interactive mode

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
classes = ["Control", "Failure"]

# Set a random seed for CPU
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Set a random seed for CUDA (if available)
if train_on_gpu:
    torch.cuda.manual_seed(seed)




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

def getDatasets(train_indices, val_indices, study_a_full_dataset):
    """Given the indices of the data in the given fold, it creates training, validation, and test datasets
    Split specs: train/val/test: 70/20/10

    Args:
        train_indices (numpy.ndarray): indices of training data of the given fold
        val_indices (numpy.ndarray): indices of validation data of the given fold
        study_a_full_dataset (torchvision.datasets.folder.ImageFolder): indices of test data of the given fold
    
    Returns:
        train_dataset (torch.utils.data.dataset.Subset): training dataset
        val_dataset (torch.utils.data.dataset.Subset): validation dataset
        test_dataset (torch.utils.data.dataset.Subset): test dataset
    """

    # Convert indices to numpy arrays
    train_indices_np = np.array(train_indices)
    val_indices_np = np.array(val_indices)
    # Convert study_a_full_dataset.targets to a NumPy array
    targets_np = np.array(study_a_full_dataset.targets)

    # Split the training indices into train and test indices
    train_indices_split, test_indices, _, _ = train_test_split(
        train_indices_np,
        targets_np[train_indices_np].ravel(),
        test_size=0.125,
        random_state=42,
        stratify=targets_np[train_indices_np].ravel()
    )

    # Create the training and testing datasets using Subset
    ## 70-20-10 split
    train_dataset = Subset(study_a_full_dataset, train_indices_split)
    val_dataset = Subset(study_a_full_dataset, val_indices)
    test_dataset = Subset(study_a_full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def test_model(model, test_loader, criterion):
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


def train_model(study_a_data_path, study_b_data_path, #MODEL SAVE #model,criterion, optimizer, scheduler, 
                epoch_list, batch_sizes, model_output, model_last_save, resume=0, resume_fold=0): #MODEL SAVE
    # Define data transformations for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define data transformations for validation
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    study_a_full_dataset = datasets.ImageFolder(root=study_a_data_path, transform=transform_train)


    
    
    # Number of folds for cross-validation
    num_folds = 5  # You can adjust this based on your needs
    # Use StratifiedKFold to maintain class distribution in each fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Assuming full_dataset is the dataset you want to perform cdu -h --max-depth 1 cross-validation on
    # You may replace it with the actual dataset variable you are using
    for fold, (train_indices, val_indices) in enumerate(skf.split(study_a_full_dataset.imgs, study_a_full_dataset.targets)):
        print("-" * 20)
        print(f"Fold {fold + 1}/{num_folds}")
        print("-" * 20)

        #MODEL SAVE
        classes = ["Control", "Failure"]
        ## Define the model parameters
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2) ## replace the number of output nodes in the last fully connected layer.
        model = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        #CHANGE - changed optimized based on wandb sweep
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 10 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # print(model_ft)

        ## Given the train and validation indices of a fold, get the train, validation and test datasets
        ## NOTE: test_dataset is created by splitting the indices within the training dataset
        ## StratifiedKFold creates train/val split in the ratio of 80-20
        ## Within this 80% of training data, 10% is split for testing.
        train_dataset, val_dataset, test_dataset = getDatasets(train_indices, val_indices, study_a_full_dataset)

        print(f"# of frames in Entire Dataset: {len(study_a_full_dataset)}")
        print(f"# of frames in training dataset {len(train_dataset)}")
        print(f"# of frames in validation dataset {len(val_dataset)}")
        print(f"# of frames in testing dataset {len(test_dataset)}")

        with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
            results_log_file.write(f"# of frames in Entire Dataset: {len(study_a_full_dataset)}\n")
            results_log_file.write(f"# of frames in training dataset {len(train_dataset)}\n")
            results_log_file.write(f"# of frames in validation dataset {len(val_dataset)}\n")
            results_log_file.write(f"# of frames in testing dataset {len(test_dataset)}\n")

        image_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

        since = time.time()
        ## Count variable to keep track of the number of combinations in the hyper-parameter tuning
        search_count = 0
        
        # Create a figure with subplots for plotting the training and validation accuracy for various batch_sizes against the # of epochs
        num_cols = 2
        num_rows = 2
        batch_figure, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))


        #MODEL SAVE
        if resume == 1:
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
        wandb.log({"fold": fold})



        for epochs in epoch_list:
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
                    best_accuracy = 0.0

                    
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

                           
                            # Store accuracy for both training and validation
                            if phase == 'train':
                                train_accuracies.append(epoch_acc)
                                train_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
                            elif phase == 'val':
                                val_accuracies.append(epoch_acc)
                                val_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "val_loss": epoch_loss, "val_accuracy": epoch_acc})

                            # break ## break from iterating over model training and validation 
                            
                            
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

                    
                    # load best model weights
                    model.load_state_dict(torch.load(model_chpt))

                    print(f"Begin testing the model")
                    true_labels, pred_labels, test_accuracy, test_loss = test_model(model, dataloaders['test'], criterion)

                    report = classification_report(true_labels, pred_labels, target_names=classes)
                    print(report)
                    wandb.log({"report": report, "val_loss": test_loss, "val_accuracy": test_accuracy})


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
                        results_log_file.write(f"------------ lab 5FPS: BEGIN SEARCH: {search_count} ------------" + "\n")
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
    
    #CHANGE - added wandb and indentations for all inside main
    color_channel = "BGR"
        data_frame_rate = 5
        dataset_path = "../../../../../data/study_data/"
        output_directory = "../../../../../data/"

        #MODEL SAVE
        last_model_save_dir = output_directory + "lab/" + "resnet50_lab5fps_last_model_fullfolds/"

        ## Define the path for storing model outputs
        now = datetime.datetime.now()
        #CHANGE - added type of dataset to path
        lab_model_output = output_directory + "lab/" + "resnet50_neckface5fps_fullfolds_" + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'
        # online_model_output = output_directory + "online/" + "resnet50_" + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

        if not os.path.exists(lab_model_output):
            os.makedirs(lab_model_output)
        
        #MODEL SAVE
        if not os.path.exists(last_model_save_dir):
            os.makedirs(last_model_save_dir)
            resume = 0
            resume_fold = 0
            wandb.log({"resume": 0})
            wandb.log({"resume_fold": resume_fold})
        else:
            #look at the models saved - they are in the format model_FOLD_EPOCH.pth. If epoch is less than 200, save variable "RESUME" as True. Look for each fold, start from 4 to 0
            for fold in range(4, -1, -1):
                #see if model exists, regardless of epoch (any epoch numbe is fine, look for partial string match with fold number and .pth extension)
                path_start = f"{last_model_save_dir}model_{fold}_*.pth"
                if glob.glob(path_start):
                    print('MODEL EXISTS FOR FOLD', fold)
                    wandb.log({"resume": 1})
                    resume = 1
                    resume_fold = fold
                    wandb.log({"resume_fold": resume_fold})
                    break
                else:
                    resume = 0
                    resume_fold = 0
                    wandb.log({"resume": 0})
                    wandb.log({"resume_fold": resume_fold})
                    print('MODEL DOES NOT EXIST FOR FOLD', fold)
        wandb.log({"job_path": lab_model_output})
        print("started from", resume_fold)
        print("resume is", resume)


        # if not os.path.exists(online_model_output):
        #     os.makedirs(online_model_output)

        ## Define lab and online studies dataset paths
        lab_frame_data_path = dataset_path + f"lab_custom_dataset/data_prefix_{data_frame_rate}_fps/frames/"
        online_frame_data_path = dataset_path + f"online_custom_dataset/data_prefix_{data_frame_rate}_fps/frames/"




        
        epoch_list = [30,50,70]
        batch_sizes = [32,64]
        model_ft = train_model(
            study_a_data_path=lab_frame_data_path,
            study_b_data_path=online_frame_data_path,
            epoch_list=epoch_list,
            batch_sizes=batch_sizes,
            model_output=lab_model_output,
            model_last_save=last_model_save_dir,
            resume=resume,
            resume_fold=resume_fold
        )


if __name__ == "__main__":
    
    main()
