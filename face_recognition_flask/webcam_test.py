import cv2

def test_webcam():
    cap = cv2.VideoCapture(0)  # Try 1, 2, etc., if 0 doesn't work

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'q' to exit the webcam test.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('Webcam Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
