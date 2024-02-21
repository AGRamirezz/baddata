import os
import cv2
import numpy as np

files_to_ignore = ['.DS_Store']

def displayFrames(participants_numpy_files_path):
    """
    Displays the frames of the response videos after readings the pixel values from each participant folder

    Args:
        participants_numpy_files_path ('str'): path to where the '.npy' files exist containing the pixel values of the frames of the responseVideos of all participants
    """

    # Obtain the list of participants
    participants = os.listdir(participants_numpy_files_path)
    participants = sorted([folder for folder in participants if folder not in files_to_ignore])

    # Participant you want to display frames for
    participant_to_display = "2088"
    print(participants)

    # Iterate through each participant in the dataset
    for participant in participants:
        if participant != participant_to_display:
            continue

        participant_frame_path = f'{participants_numpy_files_path}{participant}'

        # Load the frame data & check its dimensions
        pixel_data = np.load(f'{participant_frame_path}/pixel_data.npy')
        print(f'\nLen of pixel_data : {len(pixel_data)}')
        print(f'pixel_data.shape : {pixel_data.shape}')
        first_frame = pixel_data[0][0]
        print(f'Shape of first frame: {first_frame.shape}')


        # Load the label data & check its dimensions
        frame_label = np.load(f'{participant_frame_path}/label_data.npy')
        print(f'\nLen of frame_label: {len(frame_label)}')
        print(f'shape of frame_label: {frame_label.shape}')
        first_label = frame_label[0]
        print(f'Shape of first_label: {first_label.shape}')
        print(f'First frame label value: {first_label}')

        # Load the participant data & check its dimensions
        participant_data = np.load(f'{participant_frame_path}/participant_data.npy')
        print(f'\nLen of participant_data: {len(participant_data)}')
        print(f'shape of participant_data: {participant_data.shape}')
        first_participant_data = participant_data[0]
        print(f'Shape of first_participant_data: {first_participant_data.shape}')
        print(f'first_participant_data value: {first_participant_data}')

        video_name_data = np.load(f'{participant_frame_path}/video_name_data.npy')
        print(f"\nLen of Video Name Data: {len(video_name_data)}")
        print(f"Shape of Video Name Data: {video_name_data.shape}")
        first_video_data = video_name_data[0]
        print(f"Len of First Video Name Data: {len(first_video_data)}")
        print(f"Shape of First Video Name Data: {first_video_data.shape}")

        print(f"Displaying frames for participant : {participant}")
        # # Display the frames of a given participant
        for i in range(len(pixel_data)):
            frame = pixel_data[i][0]
            label = frame_label[i]
            video_name = video_name_data[i]

            if video_name != "fh1":
                continue
            cv2.imshow(f'Frame: {i} | Label: {label} | Participant: {participant}', frame)
            cv2.waitKey(10)
            cv2.destroyWindow(f'Frame: {i} | Label: {label} | Participant: {participant}')
        cv2.destroyAllWindows()
        break

def main():
    desired_fps = 30
    # Define the directory where the frames that are converted to numerical values are stored at
    participants_numpy_files_path = f"../data/study_data/final_lab_study_data/baddata_frames/BGR/final_BGR_{desired_fps}fps/"
    
    # Display the frames
    displayFrames(participants_numpy_files_path)

if __name__ == "__main__":
    main()
