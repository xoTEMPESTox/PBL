import os
import face_recognition
import pickle

def encode_known_faces(known_faces_dir='known_faces'):
    known_encodings = []
    known_names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                else:
                    print(f'No faces found in {image_path}')

    # Save encodings to a file for later use
    with open('known_faces_encodings.pkl', 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

    print(f'Encoded {len(known_names)} faces.')

if __name__ == "__main__":
    encode_known_faces()
