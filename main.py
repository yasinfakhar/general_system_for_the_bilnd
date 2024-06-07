import threading
import cv2
import numpy as np
from config import Config
import time
from modules.depth_estimation import depth_estimation
from modules.face_recognition import face_recognition, get_embedding, load_embeddings, findCosineDistance
from modules.get_embedding_from_image import save_embedding_for_new_faces
from modules.age_classification import get_age_cathegory
from modules.emotion_recognition import recognize_emotion
from modules.find_objects import find_objects

save_embedding_for_new_faces()

known_people = load_embeddings()
print(f'[INFO] Known people : {[name for name in known_people.keys()]}')

cap = cv2.VideoCapture(Config.camera_source)

cv2.namedWindow('faces', cv2.WINDOW_NORMAL)
cv2.namedWindow('depth-map', cv2.WINDOW_NORMAL)
cv2.namedWindow('looking_objects', cv2.WINDOW_NORMAL)

app_mode = 'face'
prompt = 'person'

proccessed_frame = 0
last_time_seen_people = {}

print('[START] App started')

name_said = False

while cap.isOpened():

    try:

        start = time.time()

        success, img = cap.read()
        img = np.flip(img, axis=1)

        if (proccessed_frame % Config.skip_rate == 0):

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            depth_map = depth_estimation(img)

            if (app_mode == "face"):

                faces, app = face_recognition(img)

                names = []
                cos_distances = []
                age_cathegories = []
                emotions = []

                if (len(faces) == 0):
                    name_said = False

                for face in faces:

                    embeding = get_embedding(img, face)

                    min_cos_distance = 1

                    person_name_with_min_sim = 'stranger'

                    for name in known_people.keys():

                        cos_distance = findCosineDistance(
                            embeding, known_people[name])

                        if (min_cos_distance > cos_distance and cos_distance <= Config.cos_distance_threshold):

                            person_name_with_min_sim = name
                            min_cos_distance = cos_distance

                    cos_distances.append(min_cos_distance)
                    names.append(person_name_with_min_sim)
                    age_cathegories.append(get_age_cathegory(face.age))
                    emotions.append(recognize_emotion(img, face))

                rimg = app.draw_on(img, faces, names,
                                   cos_distances, age_cathegories, emotions)
                rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)

                cv2.imshow('faces', rimg)

            elif (app_mode == "seg"):

                looking_objects = find_objects(
                    img, text_prompt=prompt.split(','))

                res = cv2.resize(looking_objects.cpu().numpy(), (800, 600))
                cv2.imshow("looking_objects", res)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

            cv2.putText(depth_map, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('depth-map', depth_map)

            key_pressed = cv2.waitKey(1)
            if key_pressed == ord('q'):
                break
            elif key_pressed == ord('f'):
                app_mode = "face"
            elif key_pressed == ord('s'):
                app_mode = "seg"
            elif key_pressed == ord('p'):
                prompt = input('enter new prompt : ')

        proccessed_frame += 1

    except Exception as e:
        print(f'[ERROR] Something went wrong , error : {e}')

cap.release()
