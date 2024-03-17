import cv2 
import mediapipe as mp
import pickle
import numpy as np
import streamlit as st

def main():
    st.title("Hand Gesture Recognition")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    model_dict = pickle.load(open('model.p', 'rb'))  # Update the path accordingly

    model = model_dict['model']
    labels_dict = {0:'A', 1:'B', 2:'C'}

    cap = cv2.VideoCapture(0)  # Update the camera index if needed

    while True:
        ret, frame = cap.read()

        data_aux = []
        x_=[]
        y_=[]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            for hands_landmarks in result.multi_hand_landmarks:
                for i in range(len(hands_landmarks.landmark)):
                    x = hands_landmarks.landmark[i].x
                    y = hands_landmarks.landmark[i].y
                    z = hands_landmarks.landmark[i].z
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            prediction = model.predict([np.asarray(data_aux)])
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
