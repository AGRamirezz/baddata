import os
import time
import datetime
import traceback
import glob
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

# Configuration constants
FILES_TO_IGNORE = [".DS_Store"]
CLASSES = ["Control", "Failure"]
SEED = 42

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
plt.ion()  # interactive mode

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def plot_batch_size_accuracy(train_accuracy, val_accuracy, batch_size, axs):
    """
    Creates subplots of training & validation accuracy scores for varying batch sizes.
    
    Args:
        train_accuracy (list): Training accuracies of the model across epochs
        val_accuracy (list): Validation accuracies of the model across epochs
        batch_size (int): The batch size used for this training run
        axs (matplotlib.axes): The subplot to plot on
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
    """
    Creates training, validation, and test datasets from the given indices.
    
    Split specifications: train/val/test: 70/20/10
    - The original data is split into 80/20 (train+test/val) by StratifiedKFold
    - The 80% portion is further split into 70/10 (train/test)

    Args:
        train_indices (numpy.ndarray): Indices of training data for the current fold
        val_indices (numpy.ndarray): Indices of validation data for the current fold
        study_a_full_dataset (torchvision.datasets.folder.ImageFolder): Full dataset
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) - Subset objects for each split
    """
    # Convert indices to numpy arrays
    train_indices_np = np.array(train_indices)
    val_indices_np = np.array(val_indices)
    targets_np = np.array(study_a_full_dataset.targets)

    # Split the training indices into train and test indices (87.5/12.5 split)
    # This creates the final 70/20/10 split from the original 80/20 split
    train_indices_split, test_indices, _, _ = train_test_split(
        train_indices_np,
        targets_np[train_indices_np].ravel(),
        test_size=0.125,  # 12.5% of 80% = 10% of total
        random_state=SEED,
        stratify=targets_np[train_indices_np].ravel()
    )

    # Create the datasets using Subset
    train_dataset = Subset(study_a_full_dataset, train_indices_split)
    val_dataset = Subset(study_a_full_dataset, val_indices)
    test_dataset = Subset(study_a_full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def test_model(model, test_loader, criterion):
    """
    Evaluates the model on the test dataset.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate
        test_loader (DataLoader): DataLoader for the test dataset
        criterion (torch.nn.Module): Loss function
        
    Returns:
        tuple: (true_labels, pred_labels, accuracy, test_loss)
    """
    # Track test metrics
    test_loss = 0.0
    class_correct = list(0. for i in range(len(CLASSES)))
    class_total = list(0. for i in range(len(CLASSES)))
    
    # Store labels for classification report
    true_labels = []
    pred_labels = []
    
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Iterate over test data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item() * inputs.size(0)

            # Get predictions
            _, pred = torch.max(output, 1)

            # Compare predictions to true labels
            correct_tensor = pred.eq(labels.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())

            # Store true and predicted labels for classification report
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())

            # Calculate test accuracy for each class
            for i in range(len(labels.data)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Calculate average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}\n")

    # Print per-class accuracy
    for i in range(len(CLASSES)):
        if class_total[i] > 0:
            test_accuracy = (class_correct[i] / class_total[i]) * 100
            correct_pred = int(class_correct[i])
            class_total_sum = int(class_total[i])
            print(f"Test Accuracy of {CLASSES[i]}: {test_accuracy:.2f}% ({correct_pred}/{class_total_sum})")
        else:
            print(f"Test Accuracy of {CLASSES[i]}: N/A (no test examples)")

    # Calculate overall accuracy
    overall_test_accuracy = (np.sum(class_correct) / np.sum(class_total)) * 100
    overall_correct_preds = int(np.sum(class_correct))
    overall_class_total_sum = int(np.sum(class_total))
    print(f'\nTest Accuracy (Overall): {overall_test_accuracy:.2f}% ({overall_correct_preds}/{overall_class_total_sum})')

    return true_labels, pred_labels, (overall_test_accuracy / 100), test_loss

def train_model(study_a_data_path, study_b_data_path, epoch_list, batch_sizes, 
                model_output, model_last_save, resume=0, resume_fold=0):
    """
    Trains a ResNet50 model using cross-validation.
    
    Args:
        study_a_data_path (str): Path to the primary dataset
        study_b_data_path (str): Path to the secondary dataset (not used in current implementation)
        epoch_list (list): List of epoch counts to try
        batch_sizes (list): List of batch sizes to try
        model_output (str): Directory to save model outputs
        model_last_save (str): Directory to save checkpoints for resuming training
        resume (int): Flag to indicate whether to resume training (0=no, 1=yes)
        resume_fold (int): Fold to resume from if resume=1
        
    Returns:
        torch.nn.Module: The trained model
    """
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    study_a_full_dataset = datasets.ImageFolder(root=study_a_data_path, transform=transform_train)
    
    # Set up cross-validation
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    
    # Iterate through each fold
    for fold, (train_indices, val_indices) in enumerate(skf.split(study_a_full_dataset.imgs, study_a_full_dataset.targets)):
        print("-" * 20)
        print(f"Fold {fold + 1}/{num_folds}")
        print("-" * 20)

        # Skip folds that are before the resume_fold if resuming
        if resume == 1 and fold < resume_fold:
            print(f"Skipping fold {fold} as it is before the fold to resume from")
            continue
            
        # Initialize model for this fold
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(CLASSES))  # Replace output layer
        model = model_ft.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Get datasets for this fold
        train_dataset, val_dataset, test_dataset = getDatasets(train_indices, val_indices, study_a_full_dataset)

        # Log dataset sizes
        print(f"# of frames in Entire Dataset: {len(study_a_full_dataset)}")
        print(f"# of frames in training dataset: {len(train_dataset)}")
        print(f"# of frames in validation dataset: {len(val_dataset)}")
        print(f"# of frames in testing dataset: {len(test_dataset)}")

        with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
            results_log_file.write(f"# of frames in Entire Dataset: {len(study_a_full_dataset)}\n")
            results_log_file.write(f"# of frames in training dataset: {len(train_dataset)}\n")
            results_log_file.write(f"# of frames in validation dataset: {len(val_dataset)}\n")
            results_log_file.write(f"# of frames in testing dataset: {len(test_dataset)}\n")

        # Organize datasets
        image_datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

        # Initialize timing and tracking variables
        since = time.time()
        search_count = 0
        
        # Create figure for plotting accuracy
        num_cols = 2
        num_rows = 2
        batch_figure, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))

        # Resume training if requested
        if resume == 1 and fold == resume_fold:
            checkpoint = torch.load(f"{model_last_save}model_variables_{resume_fold}.pth")
            epoch_restart = checkpoint['epoch']
            train_losses = checkpoint['train_losses']
            train_accuracies = checkpoint['train_accuracies']
            val_losses = checkpoint['val_losses']
            val_accuracies = checkpoint['val_accuracies']
            model.load_state_dict(torch.load(f"{model_last_save}model_{resume_fold}_{epoch_restart}.pth"))
            print(f"Resuming from epoch {epoch_restart} in fold {fold}")
            resume = 0  # Reset resume flag after loading
        else:
            epoch_restart = 1
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
        # Log fold information to wandb
        wandb.log({"fold": fold})

        # Iterate through hyperparameter combinations
        for epochs in epoch_list:
            # Skip invalid epochs or if we're already past this epoch count
            if epochs == 0 or epoch_restart >= epochs:
                continue
                
            for i, batch_size in enumerate(batch_sizes):
                if batch_size == 0:
                    continue

                try:
                    search_count += 1
                    print(f"Epochs: {epochs}, Batch Size: {batch_size}")

                    # Create dataloaders with current batch size
                    dataloaders = {
                        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                                     shuffle=True, num_workers=4, pin_memory=True) 
                        for x in ['train', 'val', 'test']
                    }
                    
                    # Directory to save checkpoints
                    tempdir = model_output 
                    best_accuracy = 0.0

                    # Training loop
                    for epoch in range(epoch_restart, epochs + 1):
                        print(f"Epoch {epoch}/{epochs}")
                        print("-" * 10)

                        # Each epoch has a training and validation phase
                        for phase in ["train", "val"]:
                            print(f"Phase: {phase}")
                            if phase == "train":
                                model.train()  # Set model to training mode
                            else:
                                model.eval()   # Set model to evaluation mode
                        
                            running_loss = 0.0
                            running_corrects = 0
                            n_batch_iterations = len(dataloaders[phase])

                            # Iterate over data batches
                            batch_count = 1
                            for inputs, labels in dataloaders[phase]:
                                inputs = inputs.to(device)
                                labels = labels.to(device)

                                # Zero the parameter gradients
                                optimizer.zero_grad()

                                # Forward pass
                                with torch.set_grad_enabled(phase == 'train'):
                                    outputs = model(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = criterion(outputs, labels)

                                    # Backward + optimize only in training phase
                                    if phase == 'train':
                                        loss.backward()
                                        optimizer.step()

                                # Track statistics
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)
                                
                                # Print progress every 5 batches
                                if batch_count % 5 == 0:
                                    print(f"Epoch: {epoch}/{epochs}; Step: {batch_count}/{n_batch_iterations}")
                                batch_count += 1

                            # Step the scheduler if in training phase
                            if phase == "train":
                                scheduler.step()

                            # Calculate epoch metrics
                            epoch_loss = running_loss / dataset_sizes[phase]
                            epoch_acc = running_corrects.double() / dataset_sizes[phase]

                            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                            # Store metrics
                            if phase == 'train':
                                train_accuracies.append(epoch_acc)
                                train_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
                            elif phase == 'val':
                                val_accuracies.append(epoch_acc)
                                val_losses.append(epoch_loss)
                                wandb.log({"epoch": epoch, "val_loss": epoch_loss, "val_accuracy": epoch_acc})
                            
                            # Save model checkpoint every 2 epochs after validation
                            if phase == 'val' and epoch % 2 == 0:
                                model_chpt = os.path.join(tempdir, f'model_chpt_{fold}_{epoch}.pth')
                                torch.save(model.state_dict(), model_chpt)
                                
                                # Save training state for resuming
                                checkpoint = {
                                    'epoch': epoch,
                                    'train_losses': train_losses,
                                    'train_accuracies': train_accuracies,
                                    'val_losses': val_losses,
                                    'val_accuracies': val_accuracies,
                                    'classes': CLASSES
                                }
                                torch.save(checkpoint, os.path.join(tempdir, f'model_variables_{fold}_{epoch}.pth'))
                                
                                # Save latest model for resuming
                                torch.save(model.state_dict(), f"{model_last_save}model_{fold}_{epoch}.pth")
                                torch.save(checkpoint, f"{model_last_save}model_variables_{fold}.pth")
                                
                                # Clean up old checkpoints
                                if os.path.exists(f"{model_last_save}model_{fold}_{epoch-2}.pth"):
                                    os.remove(f"{model_last_save}model_{fold}_{epoch-2}.pth")

                        print()  # Empty line between epochs
                        
                    # Training complete for this hyperparameter combination
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best val Acc: {best_accuracy:4f}')

                    # Load best model weights
                    model.load_state_dict(torch.load(model_chpt))

                    # Test the model
                    print(f"Begin testing the model")
                    true_labels, pred_labels, test_accuracy, test_loss = test_model(model, dataloaders['test'], criterion)

                    # Generate and log classification report
                    report = classification_report(true_labels, pred_labels, target_names=CLASSES)
                    print(report)
                    wandb.log({"report": report, "val_loss": test_loss, "val_accuracy": test_accuracy})

                    # Generate confusion matrix
                    conf_matrix = confusion_matrix(true_labels, pred_labels)
                    plt.figure(figsize=(8, 8))
                    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                               xticklabels=CLASSES, yticklabels=CLASSES)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.title("Confusion Matrix")
                    
                    # Save confusion matrix
                    confusion_matrix_path = model_output + f"confusion_matrices/"
                    if not os.path.exists(confusion_matrix_path):
                        os.mkdir(confusion_matrix_path)
                    
                    confusion_matrix_file = confusion_matrix_path + f"confusion_matrix_{epoch}_{batch_size}.png"
                    plt.savefig(confusion_matrix_file)
                    plt.clf()

                    # Plot accuracy curves
                    row_idx = i // num_cols
                    col_idx = i % num_cols
                    ax = axs[row_idx, col_idx]
                    ax.clear()  # Clear previous plots
                    plot_batch_size_accuracy(train_accuracies, val_accuracies, batch_size, ax)

                    # Save batch size comparison plots
                    batch_size_results_path = model_output + 'batch_size_results/'
                    if not os.path.exists(batch_size_results_path):
                        os.makedirs(batch_size_results_path)

                    batch_size_accuracy_path = batch_size_results_path + f'batch_size_comparison_{epoch}.png'
                    batch_figure.tight_layout(pad=2.5)
                    batch_figure.savefig(batch_size_accuracy_path)
                    plt.close(batch_figure)
                    plt.clf()

                    # Log results to file
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
                        results_log_file.write(f'Confusion matrix saved at {confusion_matrix_file}' + '\n')
                        results_log_file.write(f"------------ END SEARCH ------------" + "\n")
                        results_log_file.write("\n")

                except Exception as e:
                    # Log exceptions
                    with open(f"{model_output}/results_output_log.txt", "a") as results_log_file:
                        results_log_file.write(
                            f"Search Count: {search_count} \n"
                            f"Epoch: {epoch}\n"
                            f"Batch Size: {batch_size}\n"
                            f"Exception {e} thrown during model training :- \n"
                            f"{traceback.format_exc()}\n"
                        )

    return model

def main():
    """
    Main function to run the training pipeline.
    
    This function:
    1. Sets up paths and configuration
    2. Checks for existing models to resume training
    3. Trains the model with specified hyperparameters
    
    To modify the training parameters, edit the variables in this function:
    - epoch_list: List of epoch counts to try
    - batch_sizes: List of batch sizes to try
    - data_frame_rate: Frame rate of the dataset
    - dataset_path: Path to the dataset
    - output_directory: Directory to save outputs
    """
    try:
        import wandb
        wandb.init(project="resnet50-training")
    except ImportError:
        # Create a mock wandb object if wandb is not installed
        class MockWandb:
            def log(self, *args, **kwargs):
                pass
        wandb = MockWandb()
    
    # Configuration parameters
    color_channel = "BGR"
    data_frame_rate = 5
    dataset_path = "../../../../../data/study_data/"
    output_directory = "../../../../../data/"

    # Directory for saving models to resume training
    last_model_save_dir = output_directory + "lab/" + "resnet50_lab5fps_last_model_fullfolds/"

    # Create output directory with timestamp
    now = datetime.datetime.now()
    lab_model_output = output_directory + "lab/" + "resnet50_neckface5fps_fullfolds_" + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

    # Create directories if they don't exist
    if not os.path.exists(lab_model_output):
        os.makedirs(lab_model_output)
    
    # Check if we need to resume training
    if not os.path.exists(last_model_save_dir):
        os.makedirs(last_model_save_dir)
        resume = 0
        resume_fold = 0
        wandb.log({"resume": 0})
        wandb.log({"resume_fold": resume_fold})
    else:
        # Look for existing model checkpoints to resume from
        resume = 0
        resume_fold = 0
        
        # Check each fold from highest to lowest
        for fold in range(4, -1, -1):
            path_pattern = f"{last_model_save_dir}model_{fold}_*.pth"
            if glob.glob(path_pattern):
                print(f'MODEL EXISTS FOR FOLD {fold}')
                resume = 1
                resume_fold = fold
                wandb.log({"resume": 1})
                wandb.log({"resume_fold": resume_fold})
                break
            else:
                print(f'MODEL DOES NOT EXIST FOR FOLD {fold}')
                
    wandb.log({"job_path": lab_model_output})
    print(f"Starting from fold {resume_fold}")
    print(f"Resume is {resume}")

    # Define dataset paths
    lab_frame_data_path = dataset_path + f"lab_custom_dataset/data_prefix_{data_frame_rate}_fps/frames/"
    online_frame_data_path = dataset_path + f"online_custom_dataset/data_prefix_{data_frame_rate}_fps/frames/"

    # Define hyperparameters to try
    epoch_list = [30, 50, 70]  # Number of epochs to train for
    batch_sizes = [32, 64]     # Batch sizes to try
    
    # Train the model
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
