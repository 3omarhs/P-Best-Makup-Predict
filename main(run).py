import glob
import os
import face_recognition
import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog

from capture import Capture
from test1 import imagesData, allA, a, b

# current_run = 'train'
current_run = 'run'
# input_image = 'filePicker'
input_image = 'camShot'
mode_selected = 0   # select which mode to start run with..
# mode_selected = 1
tolerance=0.80  #un-accuracy "error" percentage
encoding_images_path = 'encoded images files'
images_path_F = 'photos/womens/'        # Train From path
images_path = ''
Female_list = []


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20

# image = 'photos/dataset_brand_cut/maybeline/fdf (2) before.png'
image = 'photos/test/img.png'






class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings_f = []
        self.known_face_names_f = []
        self.frame_resizing = 0.25


    def load_encoding_images(self, images_path, category):
        print(f'{category} Images Encoding:')
        if current_run == 'train':
            for img_path in images_path:
                img = cv2.imread(img_path)  # read and show image
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to another color space, There are more than 150 color-space
                basename = os.path.basename(img_path)
                (filename, ext) = os.path.splitext(basename)  # split colored/multi-channel image into separate single-channel images
                filename = str(os.path.dirname(img_path).split('/')[-1].split('\\')[-1])
                try:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]  #return the 128-dimension face encoding for each face
                except:
                    1
                self.known_face_encodings_f.append(img_encoding)
                self.known_face_names_f.append(filename)
            with open(f'{encoding_images_path}/{category}_encoding.txt', "wb") as fp:  # Pickling
                pickle.dump(self.known_face_encodings_f, fp)

            with open(f'{encoding_images_path}/{category}_names.txt', "wb") as fp:  # Pickling
                pickle.dump(self.known_face_names_f, fp)
            self.known_face_encodings_f = []
            self.known_face_names_f = []
            with open(f'{encoding_images_path}/{category}_encoding.txt', "rb") as fp:  # Unpickling
                self.known_face_encodings_f = pickle.load(fp)
            with open(f'{encoding_images_path}/{category}_names.txt', "rb") as fp:  # Unpickling
                self.known_face_names_f = pickle.load(fp)
            print(f"{category} Encoding images loaded")

        else:
            with open(f'{encoding_images_path}/{category}_encoding.txt', "rb") as fp:  # Unpickling
                self.known_face_encodings_f = pickle.load(fp)
            with open(f'{encoding_images_path}/{category}_names.txt', "rb") as fp:  # Unpickling
                self.known_face_names_f = pickle.load(fp)
            print(f"{category} Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)  # to change photo size
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # convert image to another color space, There are more than 150 color-space
        face_locations = face_recognition.face_locations(rgb_small_frame)  # bounding boxes of human faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # return the 128-dimension face encoding for each face
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings_f, face_encoding, tolerance=tolerance)  # Compare faces to see if they match
            name = "Can`t Detect"
            face_distances = face_recognition.face_distance(self.known_face_encodings_f, face_encoding)  # get distance (un-similarity) for each comparison face
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names_f[best_match_index]
            else:
                print('unknown detected!!')
            if len(face_names) < 2:
                face_names.append(name)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names



def main_GUI():
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, (540, 380), interpolation=cv2.INTER_AREA)
    # frame = resized
    h,w,o = img.shape
    if h<250 or w<250:
        frame = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_AREA)
    else:
        frame = img
    # imgBG[170:650, 725:1365] = frame
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # creating a smaller frame for better optimization:
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        try:
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)  # to add text into an image
            # print(name)
        except:
            1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 3)  # draw rectangle on photo “usually used for face boundaries”

    cv2.waitKey(1)  # read and show image
    cv2.imshow("3omar.hs Detection..", frame)  # read and show image

if input_image == 'filePicker':
    tk.Tk().withdraw()
    image = filedialog.askopenfilename()
elif input_image == 'camShot':
    image = Capture.Image()
imagesData()
sfr = SimpleFacerec()
# sfr.load_encoding_images(images_path_F, 'Female')
for i in range(len(list(b))):
    sfr.load_encoding_images(allA, list(b)[i])
try:
    while True:
        main_GUI()
except: 1