import socket
import cv2
import numpy as np
import pyautogui
import pickle
import struct
import time

receiver_ip = '192.168.0.189'
receiver_port = 9999

screen_width, screen_height = pyautogui.size()
#region = (0, 0, screen_width, screen_height)
region = (350, 50, 980, 1000)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((receiver_ip, receiver_port))
print("Connected to receiver.")
time.sleep(1)

try:
    while True:
        screenshot = pyautogui.screenshot(region=region)
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480))

        # JPEG compress the frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, encoded_img = cv2.imencode('.jpg', frame, encode_param)

        # Send frame
        data = pickle.dumps(encoded_img)
        size = struct.pack(">L", len(data))
        client_socket.sendall(size + data)

        # Limit FPS
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping stream...")

finally:
    client_socket.close()
