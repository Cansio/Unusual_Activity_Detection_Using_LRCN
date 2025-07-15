import socket
import pickle
import struct
import threading
import cv2
import numpy as np
import time
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from twilio.rest import Client
from keras.models import load_model
from multiprocessing import Process, Queue as MPQueue, Event

# Constants
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["LiftVandalism","Violence","Normal"]
host_ip = '0.0.0.0'
port = 9999

# Alert Credentials (Keep secure!)
TWILIO_SID = "your-twilio-sid"
TWILIO_AUTH_TOKEN = "your-twilio-token"
TWILIO_PHONE_NUMBER = "your-twilio-phone"
RECIPIENT_PHONE_NUMBER = "your-phone"

SENDER_EMAIL = "sender-email"
SENDER_PASSWORD = "sender-password"
RECEIVER_EMAIL = "reciever-email"

# Multiprocessing queues and event
frame_queue = MPQueue()
result_queue = MPQueue()
stop_event = Event()

# Preprocessing
def preprocess(frame):
    resized = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized = resized / 255.0
    return normalized

# Inference Process
def inference_worker(frame_queue, result_queue, stop_event):
    frame_buffer = []
    model = load_model(r"liftv_vio_nonvio_model2025_06_19__11_59_52___Loss_0.3231___Accuracy_0.9956.keras")
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            processed = preprocess(frame)
            frame_buffer.append(processed)

            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(frame_buffer), axis=0)  # Shape: (1, 20, 64, 64, 3)

                start_time = time.time()
                predicted_labels_probabilities = model.predict(input_data)[0]
                end_time = time.time()

                predicted_class_index = np.argmax(predicted_labels_probabilities)
                predicted_class_name = CLASSES_LIST[predicted_class_index]
                confidence_score = f"{predicted_labels_probabilities[predicted_class_index]:.4f}"

                inference_time = end_time - start_time
                fps = SEQUENCE_LENGTH / inference_time if inference_time > 0 else 0

                print(f"[Prediction] {predicted_class_name} ({confidence_score})")
                print(f"Inference Time: {inference_time:.4f} seconds")
                print(f"Inference FPS: {fps:.2f}\n")

                result_queue.put(predicted_class_name)
                frame_buffer.clear()

# Alerts (Email & SMS)
def send_email_alert(image_bytes):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "Violence Detected!"
    msg.attach(MIMEText("A violent activity was detected. See attached image.", 'plain'))

    attachment = MIMEBase('application', 'octet-stream')
    attachment.set_payload(image_bytes)
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', 'attachment', filename='alert.jpg')
    msg.attach(attachment)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Email alert sent.")
    except Exception as e:
        print(f"Email alert failed: {e}")

def send_sms_alert():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="Violence detected in the video stream!",
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"SMS sent. SID: {message.sid}")
    except Exception as e:
        print(f"SMS failed: {e}")

# Receive Frames via Socket
def receive_frames(conn, addr):
    print(f"[Receiver] Connected by {addr}")
    data = b""
    payload_size = struct.calcsize(">L")
    while True:
        try:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]
            encoded_img = pickle.loads(frame_data)
            frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            frame_queue.put(frame)

        except Exception as e:
            print("[Receiver] Error:", e)
            break
    conn.close()

# Display and Act on Results
def process_frames():
    fps_counter = 0
    fps_start_time = time.time()
    last_alert_time = 0

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            cv2.imshow("Live Stream", frame)
            if not result_queue.empty():
                result = result_queue.get()
                if result != "Normal" and time.time() - last_alert_time > 5:
                    print(f"[⚠ ALERT] {result.upper()} detected at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        image_bytes = jpeg.tobytes()
                        send_email_alert(image_bytes)
                        # send_sms_alert()
                    last_alert_time = time.time()

            fps_counter += 1
            if time.time() - fps_start_time >= 1:
                print(f"[Receiver] FPS: {fps_counter}")
                fps_counter = 0
                fps_start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                frame_queue.put(None)
                stop_event.set()
                break
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    print(f"[Receiver] Listening on {host_ip}:{port}")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host_ip, port))
    server_socket.listen(1)
    conn, addr = server_socket.accept()

    inference_process = Process(target=inference_worker, args=(frame_queue, result_queue, stop_event))
    inference_process.start()

    t1 = threading.Thread(target=receive_frames, args=(conn, addr))
    t2 = threading.Thread(target=process_frames)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    inference_process.join()
    server_socket.close()
    print("[Receiver] Shutdown complete.")