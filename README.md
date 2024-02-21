# BAD Data
## A Study on Domain Generalization for Failure Detection through Human Reactions in HRI
Maria Teresa Parreira, Sukruth Gowdru Lingaraju, Adolfo Ramirez-Aristizabal, Manaswi Saha, Michael Kuniavsky, Wendy Ju


You can read the paper here: (waiting reviews)
And cite is as:




### Datasets 

1. Online
2. Lab

These datasets might be made available upon request to the researchers. Please contact Teresa (mb2554@cornell.edu)

```
Color Channels:
1. BGR
2. RGB

Frame Rates for both channels:
1. 5 FPS
2. 15 FPS
3. 30 FPS
```

The file structure is as follows:

```
├── code
│   ├──preprocess
│   │	├── badnet_createDataset.py
│   │	├── badnet_createDataset_with_frames_according_to_class_label.py
│   │	└──  badnet_read_frames_from_npy.py
│   └── model_training
│       └── resnet50
│           ├── mixed_participants
│           └── non_mixed_participants
│               └── resnet50_nonmixed_online_15fps.py
│              
├── data
│   └── study_data
│       ├── final_online_study_data
│       │   └── baddata_frames
│       │       └── BGR
│       │           ├── final_BGR_30fps
│       │           └──  final_BGR_30fps.zip
│       ├── final_lab_study_data
│       │   └── baddata_frames
│       │       └── BGR
│       │           ├── final_BGR_30fps
│       │           └── final_BGR_30fps.zip
│       └── online_custom_dataset
│           └── data_prefix_30_fps
│               ├── frames
│               │   ├── 0
│               │   └── 1
│               └── meta
│                   └── metadata.txt
├── README.md
├── requirements.txt
```

Functionalities of the files are as follows:

2. `requirements.txt`: library dependencies required for training the model with the use of dataLoaders to read the frames from memory. Here, the model training is based off of PyTorch.

#### /data/study_data

Contains participant data for the `Online` and `Lab` studies.
Each study has participant frames in `BGR` and `RGB` Color channels of varying `frame_rates` as mentioned above in their respective directories.  

On each folder -- final_{lab,online}_study_data -- there would be zip files containing the dataset.

- The `.zip` files are extracted using the `badnet_createDataset.py` or `badnet_createDataset_with_frames_according_to_class_label.py` scripts.

Each of these `.zip` files contains `participant` directories (final shortlisted participants to be considered in the study):
- Here, the files that are present are as follows:
```
1. label_data.npy : frame's class label to the participant's response
2. participant_data.npy : participant ID value for the corresponding frame
3. pixel_data.npy : participant's response frame (pixel values)
4. video_name_data.npy :participant's response mapping to the corresponding stimulus video viewed
```

Dataset needs to be partitioned into the respective classes to which the frames belong to. Here, it would be `0` for neutral reaction and `1` for failure. They are stored under `data/study_data/{studyName}_custom_dataset/`. There exists a metadata file - information on frames and their class, participant, stimulusVideo that they belong to and the directory they are stored under.

#### /code

1. `badnet_createDataset.py`:  reads the `{study_name}`participant response videos - at a defined `frame_rate` and assigns class labels into:
```
control: 0
failure: 1
```
based on the timestamp of the failure occurrence and stores them into a `.npy` file.  


2. `badnet_createDataset_with_frames_according_to_class_label.py`: performs the same action as the `badnet_createDataset.py` - except that it considers participant frames only from the moment of failure and discards all the frames before the failure occurrence.  
  
3. `badnet_read_frames_from_npy.py`: is used to check and display if the extracted participant frames into `.npy` files are accurate or not.

4. `model_training/resnet50/baddata_resnet50_model_training.py`: this is the model fine-tuning script based off on the ResNet50 model architecture of ImageNet.

5. `model_training/resnet50/training_versions/`: consists of scripts that can be used to train the model on `Lab` or `Online` study using `5, 15, and 30 FPS` datasets.


## Steps to run on cluster without DataLoaders:

1. When training on the cluster - upload the relevant `.zip` dataset files from both the studies as per the directory specification shown above and `unzip` the files (These datasets are present on the Box).
2. On the GYM, due to System and GPU memory limits, considering more than 1 participant in each fold results in memory leaks and causes the model to crash.  
As a result, the `train`, `validation`, and `test` split size in each fold of both the studies are set `=1`.
3. Increasing the `batch_size` above `32` also results in memory leaks.
4. On the cluster, change the above parameters as required. If more GPUs are available (for faster training) - specify the GPU devices to be considered within the `train_model()` within the `model_training/resnet50/baddata_resnet50_model_training.py` script.

NOTE:  

1. Executing the `badnet_createDataset.py` or `badnet_createDataset_with_frames_according_to_class_label.py` is only required if the study frame files dataset is not available and only the study participant video dataset is present.
2. Since we are performing domain generalization - both the datasets can be swapped for training and testing.
As a result, `study_a_participants` refer to the study we are using for training and validation, and `study_b_participants` refer to the study data we are using for testing as required.
- This can be interchanged for training the model on either studies - by passing in the relevant study_data_directory in the `main()` method for `training_participant_data_path` and `testing_participant_data_path` parameters for the `train_model()` method.
- Based on this, you would need to define the relevants splits for the studies.


## Steps to run:

1. Create a virtual environment - the requirements file is at `requirements.txt`.
2. Make sure the files under `code/model_training/resnet50/` are present as shown in the tree above.
3. Run the `createDataset.py` for the required study - `lab` and/or `online` - and change the `data_frame_rate` parameter value in the `main()` function to create dataset of varying frame rates.
- This requires the participant frames for reading and creating the class hierarchy. Make sure the frames are under `/data/study_data/final_{studyName}_study_data/baddata_frames/{color_channel}/` as shown above in the tree.
4. Run the `code/model_training/resnet50//training_versions/resnet50_{studyName}_{frame_rate}_fps.py` file to train the model for the required study with its corresponding dataset.

NOTE: 

1. During fine-tuning, the dataset is split into train/validation/test sets as per 70/20/10 ratio.
2. Make sure to remove the `break` statments within the `train_model()` function as required.
3.  Make sure to change the `batch_sizes` as required - based on max batch_size the system is capable of processing the data.
