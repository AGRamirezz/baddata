import re
import os
import ast
import cv2
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

files_to_ignore = ['.DS_Store', 'Extracted Data']

def getData(img, label):
    """_summary_

    Args:
        img ('numpy.ndarray'): pixel values of the video frame
        label ('int'): class to which the frame belongs to

    Returns:
        X_data: pixel values of the video frame after resizing and conversion of the image channels
        y_data ('numpy.ndarray'): of class label
    """

    # # Define the height and width of the video frame (image) to be resized to
    desired_width = 224
    desired_height = 224

    X_data = []
    y_data = []

    try:
        # Get the dimensions of the image
        height, width, _ = img.shape # cv2 image has the dimensions as (height, width, #channels)
        # print('Original Image width, and height = : ', width, height)
        # print('Original Image shape: ', img.shape)

        # Calculate the side length for the square crop (minimum of width and height)
        side_length = min(width, height)

        # Calculate the coordinates for cropping the square from the center
        left = (width - side_length) // 2
        top = (height - side_length) // 2
        right = left + side_length
        bottom = top + side_length

        # Crop the image to a square to generate an image having 1:1 Aspect ratio
        square_img = img[top:bottom, left:right]
        # print('Shape of the image after centre cropping it to aspect ratio:', square_img.shape)

        # Resize the square_img to the desired dimensions
        resized_img = cv2.resize(square_img, (desired_width, desired_height))
        # print(resized_img)
        # print('Shape of the image after resizing the image down to desired height and width', resized_img.shape)

        # Convert the image from BGR to RGB color encoding format (optional: if not needed, remove the '#' in the below line)
        rgb_image = resized_img # [..., : : -1]
        # print(rgb_image.shape)

        X_data.append(rgb_image)
        y_data.append(label)

    except Exception as e:
        print('Exception {e} thrown for :- ')
        traceback.print_exc()

    return np.array(X_data), np.array(y_data)

def processFrames(desired_fps=30, participants_studyData_path='', participant_info_path='', output_path='./badnet_data/'):
    """
    This method obtains the path to the studyData of the participants's response videos
    It reads & processes all the participant video frames and stores them as a 'numpy.ndarray' value file

    Args:
        participants_studyData_path ('str'): path where the studyData of the participants exists (i.e: response videos of the participants)
        participant_info_path (_type_): path where the metadata of the participants exist
        output_path (_type_): path where the '.npy' files is to be stored
    """

    if participant_info_path != '':
        ### For online
        # # Read the participant metadata log file
        # participant_info_files =  os.listdir(participant_info_path)
        # participant_log_df = pd.read_excel(f'{participant_info_path}participant_log.xlsx')

        # # Participants that are to be considered
        # final_participants = []

        # # From the participant log file, consider only the participants who are to be included in the study
        # for indx, row in participant_log_df.iterrows():
        #     if row['Include as Study Data?'] == 'Y':
        #         final_participants.append(int(row['Participant']))
        
        ### For lab
        ## Read the participant metadata log file
        participant_info_files =  os.listdir(participant_info_path)
        participant_log_df = pd.read_csv(f"{participant_info_path}/participant_list.csv")
        
        # Participants that a/re to be considered
        final_participants = []

        # From the participant log file, consider only the participants who are to be included in the study
        for indx, row in participant_log_df.iterrows():
            if row["study_type"] == "main_study":
                final_participants.append(row["qualtrics_id"])
    else:
        final_participants = [participant for participant in os.listdir(participants_studyData_path) if participant not in files_to_ignore]

    final_participants = sorted(final_participants)
    print('Final Participants from the Log File = \n', final_participants)
    print('Total Participants from the Log File = ', len(final_participants))
    
    # A dictionary of video class labels 
    response_class_dict = {
        'ch': ['control', 0],
        'cr': ['control', 0],
        'fh': ['failure_human', 1],
        'fr': ['failure_robot', 2]
    }

    # # If all frames for all the videos of all participants are to be stored to a single '.npy' file
    # all_pixel_data = []
    # all_label_data = []
    # all_participant_data = []

    for participant in tqdm(final_participants, total=len(final_participants), desc='Processing Participants: '):

        # Define the path where the '.mp4' videos of the participants are stored that are to be considered
        participant_path = f'{participants_studyData_path}{participant}/mp4StudyVideo/'
        participant_videos = sorted([video for video in os.listdir(participant_path) if video not in files_to_ignore]) # Filter to retain only .mp4 files to be considered
        # print(participant_videos)
        # break

        # Specify the directory where the '.npy' files of the frames from the videos are to be saved at
        participant_output_path = f'{output_path}{participant}/'
        if not os.path.exists(participant_output_path):
            os.mkdir(participant_output_path)

        try:
            # # NOTE: 
            # # If you want to store 'frames of all videos of all participants' and write them directly to a single '.npy' local file,
            # # move the initialisation of these variables to be before the 'participant' loop - i.e: 'all_pixel_data', 'all_label_data', 'all_participant_data'.
            # # You may run into memory leak errors - as the data for processing all the frames of all the videos of all participants is great in memory size
            # # To overcome that, we store the data to local file after iterating through 'each' participant
            
            # # For a given participant, iterate through all their videos and store:
            # # - the frames to 'pixel_data'
            # # - the label of those frames to 'label_data'
            # # - the participant ID to 'participant_data'
            
            pixel_data = []
            label_data = []
            participant_data = []
            i = 0 # To keep track of the frame number
            for video in participant_videos:    # For the given participant, iterate through all their videos

                ### The below line assumes that the response videos have been renamed to the class label that they belong to (Projects - online/lab)
                ### and that the response videos renaming includes the video count of how many times the stimulusVideo has been replayed
                if video.split('_')[1].split('.')[0] == '1':    # consider only the first video recording response for the stimulus video
                    video_path = f'{participant_path}{video}'
                    # print(video_path)

                    # Obtain the names and labels of the video file
                    video_name = video.split('_')[0]
                    video_class = response_class_dict[re.sub(r'[^a-zA-Z]', '', video_name)][0] # basically slice the name of video file to bring it down to the dict key format
                    video_label = response_class_dict[re.sub(r'[^a-zA-Z]', '', video_name)][1]

                    # Start reading the video file
                    cap = cv2.VideoCapture(video_path)
                    ret = True

                    # # Define the required 'fps' rate if any
                    current_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the current FPS from the VideoCapture object
                    # cap.set(cv2.CAP_PROP_FPS, desired_fps)  # Set the FPS for the VideoCapture object, if you desire it to be different
                    # current_fps = cap.get(cv2.CAP_PROP_FPS)
                    # print(f'Participant {participant} video being captured at fps = ', desired_fps)
                    interval = int(current_fps / desired_fps)

                    frame_counter = 0
                    # # While there 'exists' frames in the video, start reading individual frames
                    while ret:
                        ret, img = cap.read()   # if there exists a frame, set 'ret' to be True and 'img' to load the frame

                        if ret:
                            ### Process frames only at the desired frame rate
                            ### The cv2.VideoCapture().set(cv2.CAP_PROP_FPS, desired_fps) - does not work as the video's fps is set
                            ### Only way to obtain frames at a desired fps is to wait until every X frame (X = desired frame rate) or skip over all the frames until you reach the Xth frame
                            ### Here we skip over the frames
                            ### Useful resources: 
                            ## - https://answers.opencv.org/question/214375/changing-the-fps-of-a-video/
                            ## - https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2/54444910#54444910
                            if frame_counter % interval == 0:
                                X_data, y_label = getData(img, video_label) #pass the frame image, to perform resizing operations

                                # Append the data accordingly
                                pixel_data.append(X_data)
                                label_data.append(y_label)
                                participant_data.append(participant)

                                # print(f'Pixel values of frame : \n{pixel_data[i][0]}')
                                # print(f'Class of frame belonging to : {label_data[i]}')
                                # print(f'Participant : {participant_data[i]}')

                                # Display the frame
                                # cv2.imshow(f'Frame {y_label} | {i}', pixel_data[i][0]) # 'i' represents the frame number
                                # cv2.waitKey(1) # Maintain a delay of 1ms before displaying the next frame
                            i += 1
                            frame_counter += 1
                            # break   # break after processing a single frame of a video

                    # cv2.destroyAllWindows() # Close all the frame windows
                    cap.release()   # Close the video file
                    # print(f'Participant: {participant}, Video: {video_name}, Class = {video_class}, Label = {video_label}, Number of Frames = {len(pixel_data)}, Number of Labels = {len(label_data)}')

                # break # break after processing a single video

            # # Write to file after processing each responseVideo of the participant in 'append' mode and reset the pixel_data variable to be = empty to prevent memory overleak
            with open(f'{participant_output_path}pixel_data.npy', 'wb') as f:
                np.save(f, np.array(pixel_data))
            with open(f'{participant_output_path}label_data.npy', 'wb') as f:
                np.save(f,  np.array(label_data))
            with open(f'{participant_output_path}participant_data.npy', 'wb') as f:
                np.save(f,  np.array(participant_data))

            print(f'Saved the numpy files for participant {participant} where the number of frames in the video = {len(pixel_data)}')

        except Exception as e:
            print(f'Exception {e} thrown for :-')
            traceback.print_exc()
            pass

        # break   # break after processing a single participant

def main():
    # CHANGE
    # Define the data directories
    participants_studyData_path = '../data/study_data/lab_study_data/survey_data/' # Path where the participant's studyData files are located
    participant_info_path = '../data/study_data/lab_study_data/metadata/' # Path where the participant metadata is stored

    desired_fps = 5 #CHANGE

    # Define the output paths
    output_path = f'../data/study_data/lab_study_data/badnet_data/BGR/BGR_{desired_fps}fps/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    processFrames(
        desired_fps=desired_fps,
        participants_studyData_path=participants_studyData_path,
        participant_info_path=participant_info_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
