import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
olho_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Captura cada um dos frames
    ret, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                minNeighbors=5)
    detections_olho = olho_detector.detectMultiScale(image_gray,minSize=(50, 50),minNeighbors=5)

    # Desenha o retangulo em volta das faces e dos olhos
    for (x, y, w, h) in detections:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in detections_olho:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # mostra o frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libera a captura de video
video_capture.release()
cv2.destroyAllWindows()