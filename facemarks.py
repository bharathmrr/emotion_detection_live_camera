import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

# Load emotion classification model
model = load_model("emotion_model_small.h5")

# Define emotion labels (update if needed)
emotion_labels = ['angry', 'disgusted','fearful', 'happy', 'sad',]

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Start webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection, \
     mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)
        mesh_results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                # Crop face for prediction
                face = frame[y:y+h_box, x:x+w_box]
                try:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    face_normalized = face_resized / 255.0
                    face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1, 48, 48, 1)

                    # Predict emotion
                    prediction = model.predict(face_input)[0]
                    emotion = emotion_labels[np.argmax(prediction)]
                    confidence = np.max(prediction)

                    # Draw emotion and box
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except:
                    pass  # Handle crop errors

        # Draw face landmarks
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        cv2.imshow("Emotion Detection + Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
