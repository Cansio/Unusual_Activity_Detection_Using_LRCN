from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
LRCN_model = load_model(
    r"C:\Users\joeca\PycharmProjects\GPU_usage\myenv\joe_model_1_2025_04_30__11_56_36___Loss_0.0939___Accuracy_0.9950.keras",
    compile=False
)
print("Model loaded successfully!")
print(LRCN_model.summary())

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["NonViolence", "Violence"]

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    print(f"Processing video: {video_file_path}")

    def preprocess_video(video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()

        while success:
            if frame is None:
                break

            # Resize and normalize (no cropping)
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames.append(normalized_frame)

            success, frame = cap.read()

        cap.release()

        if len(frames) < SEQUENCE_LENGTH:
            print(f"Not enough frames: Only {len(frames)} available")
            return None

        return frames

    # Preprocess the video
    preprocessed_frames = preprocess_video(video_file_path)

    if preprocessed_frames is None:
        print("Video preprocessing failed.")
        return

    # Show the first frame
    first_frame_display = (preprocessed_frames[0] * 255).astype(np.uint8)
    cv2.imshow("First Frame (Resized)", first_frame_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Prepare input for model
    input_sequence = np.array(preprocessed_frames[:SEQUENCE_LENGTH]).reshape(1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    print(f"Input Shape: {input_sequence.shape}")
    prediction = LRCN_model.predict(input_sequence)[0]

    predicted_class = CLASSES_LIST[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]

    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

# Run prediction
input_video_file_path = r"C:\Users\joeca\PycharmProjects\GPU_usage\myenv\stationary cam test videos\v13.mp4"
predict_single_action(input_video_file_path, SEQUENCE_LENGTH)