import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_con=0.5, tracking_con=0.5, model_path='models/model1'):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_con, self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils

        # Load the TensorFlow model
        self.model = tf.keras.models.load_model(model_path)

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return landmark_list

    def classify_asl_letter(self, img, landmark_list):
        if len(landmark_list) == 0:
            return None  # No hand detected

        # Prepare the image for prediction
        preprocessed_img = self.extract_features(img)

        # Predict the letter
        predicted_letter = self.model.predict(np.array([preprocessed_img]))

        return predicted_letter

    def extract_features(self, img):
        # Resize the image to match the model's expected input size
        resized_img = cv2.resize(img, (150, 150))
        # Convert image to grayscale
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Add channel dimension to match expected input shape
        gray_img = np.expand_dims(gray_img, axis=-1)
        # Normalize the pixel values
        normalized_img = gray_img / 255.0
        return normalized_img

    def decode_predictions(self, predictions):
        predicted_index = np.argmax(predictions)
        if predicted_index < 26:
            predicted_letter = chr(predicted_index + 65)
        elif predicted_index == 26:
            predicted_letter = 'Space'
        elif predicted_index == 27:
            predicted_letter = 'Delete'
        else:
            predicted_letter = 'Nothing'
        return predicted_letter

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    last_asl_letter = None
    last_detection_time = time.time()
    detection_interval = 1.0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=False)

        if len(landmark_list) != 0:
            # Pass the whole image for classification
            asl_letter = detector.classify_asl_letter(img, landmark_list)
            current_time = time.time()
            if asl_letter is not None:
                asl_letter_str = detector.decode_predictions(asl_letter)
                if asl_letter_str != last_asl_letter or (current_time - last_detection_time) >= detection_interval:
                    print(f"Predicted ASL Letter: {asl_letter_str}")
                    last_asl_letter = asl_letter_str
                    last_detection_time = current_time

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
