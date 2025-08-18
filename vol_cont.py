import time #For displaying frame rate (FPS)
import cv2 #Captures and handles image
import pyautogui #Give inputs to computer
import mediapipe as mp #Recognize hand gestures

# Initialize video capture
capture = cv2.VideoCapture(0) #index of the camera in computer, (1 camera only so index is "0")

# Initialize mediapipe and pause model which will be responsible to recognize hand gestures
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create a drawing utility object, useful to draw hand landmarks on the frames
mp_drawing = mp.solutions.drawing_utils

# For calculating FPS
previous_time = 0

# Defaults to False
volume_control_enabled = False

# Read the frames and process each frame which the hand gestures
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # cv2.flip(image, flipCode)
    frame = cv2.flip(frame, 1)

    # Convert each frame to RGB (required by mediapipe library)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Now we can process the frame with mediapipe library
    results = hands.process(image_rgb)

    # To recognize if there is hand landmarks in the processed image
    # Check if mediapipe detected some gestures
    if results.multi_hand_landmarks:
        # For every hands landmarks in the camera
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Set the volume-controlling point (tip of index finger) and the base point that they need to cross 
            # ".y" at the end indicates that is it using the y-coordinates
            # The screen works in a set of top-left coordinate (0, 0), normalized to frame height, not pixel coordinates
            #   - Top: 0
            #   - Bottom: 1
            #   - Left: 0
            #   - Right: 1
            controller_point_1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y 
            controller_point_2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            base_point_1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            base_point_2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            
            # Logic for gestures using coordinates
            if volume_control_enabled:
                if controller_point_1_y < base_point_1_y and controller_point_2_y < base_point_2_y:
                    hand_gesture = "pointing up"
                elif controller_point_1_y > base_point_1_y and controller_point_2_y > base_point_2_y:
                    hand_gesture = "pointing down"
                else:
                    hand_gesture = "others"

                # Use .press() from pyautogui module to send a virtual key press from the OS
                if hand_gesture == "pointing up":
                    pyautogui.press("volumeup")
                elif hand_gesture == "pointing down":
                    pyautogui.press("volumedown")

    # Calculate FPS
    current_time = time.time() # Returns the current time in seconds as a float
    fps = 1 / (current_time - previous_time) # (current_time - previous_time) calculates the time difference between current frame and previous frame, 1/ (...) converts time per frame to frames per second
    previous_time = current_time #Updates previous time for the next iteration of the loop

    # Display FPS on the frame
    # "frame": the image/frame where the text will be drawn
    # "(10, 30)": (x, y) position in pixels from top-left corner of the frame
    # "0.5": fontScale
    # "2": thickness of the text lines in pixels
    # cv2.putText(image/frame, text string, (x,y) positioning, font, fontScale, text color, text thickness)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Code for camera window popup, showing webcam feed
    # cv2.imshow("Name", "frame of webcam")
    # cv2.imshow() must be called after modifying the frame, or else changes wont appear
    cv2.imshow("Hand Tracking", frame)

    # Keeps the window responsive and check for key presses, press q to quit program
    # cv2.waitKey(1):
    #   - waits 1ms for a key press
    #   - returns ASCII code of key pressed
    #   - "1ms" also lets OpenCV refresh the window. Without waitKey, the window will freeze or not appear
    # & 0xFF:
    #   - 0xFF is hexadecimal for 11111111 in binary
    #   - The bitwise AND operator "&" compares each bit
    #   - Ensures compability across platforms by keeping only the last 8 bits of the key code.
    #   - Needed on some systems for correct key detection.
    key = cv2.waitKey(1) & 0xFF
    # volume_control_enabled (False initially) will be not (False), which will become True
    # "volume_control_enabled = not volume_control_enabled" switches volume_control_enabled between True and False
    #   - if "v" key is pressed, it starts switching
    if key == ord('v'):
        volume_control_enabled = not volume_control_enabled
    # ord('q'):
    #   - Converts the character 'q' to its ASCII value
    elif key == ord('q'):
        break

# Release camera and close windows after loop ends
# capture.release():
#   - Frees the camera resource so other programs can use it
capture.release()
# cv2.destroyAllWindows():
#   - Closes all OpenCV windows opened by cv2.imshow()
cv2.destroyAllWindows()