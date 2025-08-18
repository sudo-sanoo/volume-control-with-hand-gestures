import time
import cv2 
import pyautogui
import mediapipe as mp

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

previous_time = 0

volume_control_enabled = False

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            controller_point_1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y 
            controller_point_2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            base_point_1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            base_point_2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            
            if volume_control_enabled:
                if controller_point_1_y < base_point_1_y and controller_point_2_y < base_point_2_y:
                    hand_gesture = "pointing up"
                elif controller_point_1_y > base_point_1_y and controller_point_2_y > base_point_2_y:
                    hand_gesture = "pointing down"
                else:
                    hand_gesture = "others"

                if hand_gesture == "pointing up":
                    pyautogui.press("volumeup")
                elif hand_gesture == "pointing down":
                    pyautogui.press("volumedown")

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('v'):
        volume_control_enabled = not volume_control_enabled
    elif key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()