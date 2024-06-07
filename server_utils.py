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

app_mode = 'face'
prompt = 'person'

proccessed_frame = 0
last_time_seen_people = {}

print('[START] ML Server started')


def inference(image, mode, **kwargs):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    depth_map = depth_estimation(image)
    depth_map = cv2.resize(depth_map, (640,480))

    if (mode == "face"):

        faces, app = face_recognition(image)

        names = []
        cos_distances = []
        age_cathegories = []
        emotions = []

        for face in faces:

            embeding = get_embedding(image, face)

            min_cos_distance = 1
            person_name_with_min_sim = 'stranger'

            for name in known_people.keys():

                cos_distance = findCosineDistance(embeding, known_people[name])

                if (min_cos_distance > cos_distance and cos_distance <= Config.cos_distance_threshold):
                    person_name_with_min_sim = name
                    min_cos_distance = cos_distance

            cos_distances.append(min_cos_distance)
            names.append(person_name_with_min_sim)
            age_cathegories.append(get_age_cathegory(face.age))
            emotions.append(recognize_emotion(image, face))
        
        rimg = app.draw_on(image, faces, names, cos_distances,
                           age_cathegories, emotions)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2BGR)

        rimg = cv2.addWeighted(rimg, 0.5, depth_map, 0.5, 0)

        return rimg

    elif (mode == "seg"):

        looking_objects = find_objects(
            image, text_prompt=kwargs['prompt'].split(','))
        res = cv2.resize(looking_objects.cpu().numpy(), (800, 600))
        res[res > 0.2] = 255
        res[res <= 0.2] = 0

        return res
