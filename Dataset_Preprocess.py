import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import shutil

seed_constant = 25
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#Create a Matplotlib figure
plt.figure(figsize = (20, 20))

dataset_dir = r"C:\Users\joeca\PycharmProjects\GPU_usage\myenv\300_dataset_lv_v_nv"

# Get the names of all classes
all_classes_names = os.listdir(dataset_dir)

all_classes_names = [name for name in all_classes_names if os.path.isdir(os.path.join(dataset_dir, name)) and not name.startswith('.')]

# Ensure num_samples <= len(all_classes_names)
num_samples = min(len(all_classes_names), 20)
random_range = random.sample(range(len(all_classes_names)), num_samples)

for counter, random_index in enumerate(random_range, 1):

  selected_class_Name = all_classes_names[random_index]

  video_files_names_list = os.listdir(f'{dataset_dir}/{selected_class_Name}')

  if not video_files_names_list:
    print(f"Skipping '{selected_class_Name}' as it contains no video files.")
    continue

  # Randomly select a video file
  selected_video_file_name = random.choice(video_files_names_list)

  # Initialize a Video object to read
  video_reader = cv2.VideoCapture(f'{dataset_dir}/{selected_class_Name}/{selected_video_file_name}')

  # Reading 1st frame of video
  _, bgr_frame = video_reader.read()

  # Release the Video object
  video_reader.release()

  # Convert the frame from BGR into RGB format
  rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

  cv2.putText(rgb_frame, selected_class_Name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

  plt.subplot(5, 4, counter);plt.imshow(rgb_frame);plt.axis('off')

  IMAGE_HEIGHT, IMAGE_WIDTH = 64,64
  SEQUENCE_LENGTH = 20
  DATASET_DIR = r'C:\Users\joeca\PycharmProjects\GPU_usage\myenv\300_dataset_lv_v_nv'
  CLASSES_LIST = ["LiftVandalism","Violence","NonViolence"]

  def frames_extraction(video_path):

      frames_list = []

      video_reader = cv2.VideoCapture(video_path)

      video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

      skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

      for frame_counter in range(SEQUENCE_LENGTH):

          video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

          success, frame = video_reader.read()

          if not success:
              break

          resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

          normalized_frame = resized_frame / 255

          frames_list.append(normalized_frame)

      video_reader.release()

      return frames_list

def create_dataset():

  features =[]
  labels = []

  for class_index, class_name in enumerate(CLASSES_LIST):

    print(f'Extracting Data of Class: {class_name}')

    files_list = os.listdir(os.path.join(DATASET_DIR, class_name))

    for file_name in files_list:

      video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

      frames = frames_extraction(video_file_path)

      if len(frames) == SEQUENCE_LENGTH:

        features.append(frames)
        labels.append(class_index)

  features = np.asarray(features)
  labels = np.array(labels)

  return features, labels

features, labels = create_dataset()

#saving the dataset in .npz format
np.savez(r'C:\Users\joeca\PycharmProjects\GPU_usage\myenv\300_lv_v_nv_dataset.npz', features=features, labels=labels)
print("Dataset saved ")
