import cv2
import mediapipe as mp
import numpy as np
import os  # For file handling

def remove_background(image):
    """
    Remove the background of an image and add transparency based on color thresholding.
    """
    if image.shape[2] == 4:  # If image has an alpha channel (RGBA)
        b, g, r, alpha = cv2.split(image)
    else:  # If image has no alpha channel (BGR)
        b, g, r = cv2.split(image)
        alpha = np.ones_like(b) * 255  # Create an opaque alpha channel

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_background = np.array([0, 0, 180])
    upper_background = np.array([255, 30, 255])
    mask = cv2.inRange(hsv_image, lower_background, upper_background)
    mask_inv = cv2.bitwise_not(mask)

    return cv2.merge((b, g, r, mask_inv))  # Add alpha transparency based on the inverse mask

# Preprocess glasses image
GLASSES_FOLDER = "glasses"
glasses_images = []
for filename in sorted(os.listdir(GLASSES_FOLDER)):
    if filename.endswith(".png"):
        image_path = os.path.join(GLASSES_FOLDER, filename)
        original_glasses = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Check if the image is loaded successfully
        if original_glasses is None:
            print(f"Error loading image: {image_path}")
            continue  # Skip this file if it can't be loaded
        
        glasses_images.append(remove_background(original_glasses))

current_glasses_index = 0
glasses_image = glasses_images[current_glasses_index]

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def overlay_rotated_image(background, overlay, position, size, angle):
    """
    Overlay a rotated image (with alpha) onto a background image at a given position.
    """
    # Resize overlay image
    overlay_resized = cv2.resize(overlay, size)
    h, w = overlay_resized.shape[:2]

    # Rotate the image
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(overlay_resized, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Overlay image with alpha blending
    x, y = position
    for i in range(h):
        for j in range(w):
            if x + j >= background.shape[1] or y + i >= background.shape[0]:
                continue
            alpha = rotated_overlay[i, j, 3] / 255.0  # Alpha channel
            if alpha > 0:  # Blend only non-transparent pixels
                background[y + i, x + j] = (
                    1 - alpha
                ) * background[y + i, x + j] + alpha * rotated_overlay[i, j, :3]
    return background

def main():
    global glasses_image, current_glasses_index
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get key landmarks: left eye, right eye, and nose tip
                left_eye = face_landmarks.landmark[33]  # Left eye corner
                right_eye = face_landmarks.landmark[263]  # Right eye corner
                nose_bridge = face_landmarks.landmark[6]  # Nose bridge landmark

                # Convert normalized coordinates to pixel coordinates
                left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))
                nose_bridge_coords = (int(nose_bridge.x * w), int(nose_bridge.y * h))

                # Calculate midpoint between eyes
                eye_center_x = (left_eye_coords[0] + right_eye_coords[0]) // 2
                eye_center_y = (left_eye_coords[1] + right_eye_coords[1]) // 2

                # Adjust vertical position to sit slightly above the nose bridge
                vertical_offset = nose_bridge_coords[1] - eye_center_y
                adjusted_center = (eye_center_x, eye_center_y + vertical_offset // 2)  # Updated calculation

                #rotation angle (eye tilt)
                delta_y = right_eye_coords[1] - left_eye_coords[1]
                delta_x = right_eye_coords[0] - left_eye_coords[0]
                angle = np.degrees(np.arctan2(delta_y, delta_x))

                #dynamic glasses size
                eye_distance = int(np.hypot(delta_x, delta_y))
                glasses_width = int(eye_distance * 1.8) 
                glasses_height = int(glasses_width * glasses_image.shape[0] / glasses_image.shape[1])

                # Top-left corner for the overlay
                glasses_position = (adjusted_center[0] - glasses_width // 2, adjusted_center[1] - glasses_height // 2)

                # Overlay rotated glasses
                frame = overlay_rotated_image(
                    frame, 
                    glasses_image, 
                    glasses_position, 
                    (glasses_width, glasses_height), 
                    angle
                )

        # Show the webcam feed
        cv2.imshow("Virtual Glasses Overlay", frame)

        # Show the selected glasses image
        # Resize the glasses image for display in the slide window
        glass_slide = cv2.resize(glasses_image, (300, 150))  # Adjust size as needed
        cv2.imshow("Selected Glasses", glass_slide)

        # Key controls for switching glasses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("n"):  # Next glasses
            current_glasses_index = (current_glasses_index + 1) % len(glasses_images)
            glasses_image = glasses_images[current_glasses_index]
        elif key == ord("p"):  # Previous glasses
            current_glasses_index = (current_glasses_index - 1) % len(glasses_images)
            glasses_image = glasses_images[current_glasses_index]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
