import cv2
import numpy as np
from hand_detection import HandDetector  # Assuming hand_detection.py contains the HandDetector class
from gtts import gTTS
import os

def text_to_speech(text):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en')
    
    # Save the audio to a file
    tts.save("output.mp3")
    
    # Play the audio file
    os.system("afplay output.mp3")

def main():
    # Initialize the hand detector
    detector = HandDetector()

    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Find hands in the frame
        frame_with_hands = detector.find_hands(frame)
        landmark_list = detector.find_position(frame_with_hands, draw=False)

        if landmark_list:
            # Pass the frame for classification
            asl_letter = detector.classify_asl_letter(frame_with_hands, landmark_list)
            if asl_letter is not None:
                asl_letter_str = detector.decode_predictions(asl_letter)
                print(f"Predicted ASL Letter: {asl_letter_str}")

                # Convert ASL letter to speech
                text_to_speech(asl_letter_str)

        # Display the frame
        cv2.imshow("ASL Translator", frame_with_hands)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
