import os
import cv2
import face_recognition
import pickle
from flask import Flask, render_template, Response
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load known faces
try:
    with open('known_faces_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
        known_encodings = data.get('encodings', [])
        known_names = data.get('names', [])
    if not known_encodings:
        logging.warning("No known face encodings found. Please run 'encode_faces.py'.")
except FileNotFoundError:
    logging.error("The 'known_faces_encodings.pkl' file was not found. Please run 'encode_faces.py' first.")
    known_encodings = []
    known_names = []
except Exception as e:
    logging.error(f"An error occurred while loading face encodings: {e}")
    known_encodings = []
    known_names = []

# Initialize video capture (0 for default webcam)
try:
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        logging.error("Cannot open webcam. Please ensure it is connected and not used by another application.")
except Exception as e:
    logging.error(f"An error occurred while accessing the webcam: {e}")
    video_capture = None

def generate_frames():
    if not video_capture:
        logging.error("Video capture is not initialized.")
        return

    while True:
        try:
            success, frame = video_capture.read()
            if not success:
                logging.error("Failed to read frame from webcam.")
                break
            else:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert BGR to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect faces and get encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # Compare face with known encodings
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches and len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]

                    face_names.append(name)
                    logging.debug(f"Detected Face: {name}")

                # Annotate the frame with boxes and names
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame was scaled down
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame in byte format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"An error occurred during frame processing: {e}")
            continue  # Skip to the next frame instead of breaking


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not video_capture:
        logging.error("Video capture is not available.")
        return "Video capture is not available.", 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"An error occurred while running the Flask app: {e}")
    finally:
        if video_capture:
            video_capture.release()
        cv2.destroyAllWindows()
