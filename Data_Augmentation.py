# Input/output directories
input_dir = r"C:\Users\joeca\PycharmProjects\GPU_usage\myenv\Dataset"
output_dir = r"Dataset_normal_300"
os.makedirs(output_dir, exist_ok=True)

# Augmentation functions
def horizontal_flip(frame):
    return cv2.flip(frame, 1)

def adjust_brightness(frame, factor=1.2):
    return np.clip(frame * factor, 0, 255).astype(np.uint8)

def zoom(frame, factor=1.2):
    h, w = frame.shape[:2]
    nh, nw = int(h / factor), int(w / factor)
    y1 = (h - nh) // 2
    x1 = (w - nw) // 2
    cropped = frame[y1:y1+nh, x1:x1+nw]
    return cv2.resize(cropped, (w, h))

def change_speed(frames, speed_factor):
    if speed_factor > 1.0:
        return frames[::int(speed_factor)]
    else:
        new_frames = []
        for frame in frames:
            new_frames.extend([frame] * int(1 / speed_factor))
        return new_frames

def gaussian_blur(frame, kernel_size=5):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def rotate(frame, angle):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(frame, M, (w, h))

def add_shadow(frame):
    top_x, top_y = frame.shape[1] * random.uniform(0.2, 0.8), 0
    bot_x, bot_y = frame.shape[1] * random.uniform(0.1, 0.9), frame.shape[0]
    shadow_mask = np.zeros_like(frame, dtype=np.uint8)
    poly = np.array([[top_x, top_y], [bot_x, bot_y], [bot_x+100, bot_y], [top_x+100, top_y]], np.int32)
    cv2.fillPoly(shadow_mask, [poly], (50, 50, 50))
    shadowed = cv2.addWeighted(frame, 1, shadow_mask, 0.5, 0)
    return shadowed

# Define augmentations
augmentations = {
    "flip": lambda fr: [horizontal_flip(f) for f in fr],
    "brightness_up": lambda fr: [adjust_brightness(f, 1.3) for f in fr],
    "brightness_down": lambda fr: [adjust_brightness(f, 0.7) for f in fr],
    "speed_up": lambda fr: change_speed(fr, 1.2),
    "speed_down": lambda fr: change_speed(fr, 0.8),
    "blur": lambda fr: [gaussian_blur(f) for f in fr],
    "rotate_p10": lambda fr: [rotate(f, 10) for f in fr],
    "rotate_m10": lambda fr: [rotate(f, -10) for f in fr],
    "shadow": lambda fr: [add_shadow(f) for f in fr],
}

# Process each video one at a time
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
augmented_count = 0

for filename in video_files:
    filepath = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Save original
    out_path = os.path.join(output_dir, filename)
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Original saved: {out_path}")

    # Apply each augmentation
    for aug_type, aug_func in augmentations.items():
        aug_frames = aug_func(frames)
        out_fps = fps * 1.2 if aug_type == "speed_up" else (fps * 0.8 if aug_type == "speed_down" else fps)

        aug_name = os.path.splitext(filename)[0] + f"aug{aug_type}.mp4"
        aug_path = os.path.join(output_dir, aug_name)
        out = cv2.VideoWriter(aug_path, fourcc, out_fps, (w, h))
        for f in aug_frames:
            out.write(f)
        out.release()
        augmented_count += 1
        print(f"🧪 Augmented ({aug_type}) saved: {aug_path}")

print(f"\nDone! {len(video_files)} original + {augmented_count} augmented = {len(video_files) + augmented_count} total videos.")
print(f"All saved in: {output_dir}")