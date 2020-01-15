# face_recog.py

import face_recognition
import cv2
import time
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame=[]
        self.conter={}
        # Load sample pictures and learn how to recognize it.
        self.dirname = '/etc/knowns'
        #global files
        files_folder = os.listdir(self.dirname)
        for folder in files_folder:
            folder_data = os.listdir(self.dirname+"/"+folder)
            for filename in folder_data:
                name, ext = filename.split(".")
                if ext == 'jpg':
                    if not str(filename) in  self.known_face_names:
                        self.known_face_names.append(str(filename))
                    pathname = os.path.join(self.dirname+"/"+folder, filename)
                    img = face_recognition.load_image_file(pathname)
                    if len(face_recognition.face_encodings(img))<1:
                        os.remove(pathname)
                        continue
                    face_encoding = face_recognition.face_encodings(img)[0]
                    self.known_face_encodings.append(face_encoding)
            self.conter[folder]= [time.time(), time.time()]
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def new_user(self):
        folder = os.listdir(self.dirname)
        try :
            folder_name = str(max(map(int, folder))+1)
        except:
            folder_name = "1"
        rgb_small_frame = self.frame[:, :, ::-1]
        save_img = rgb_small_frame[self.face_locations[0][0] : self.face_locations[0][2],self.face_locations[0][3]:self.face_locations[0][1]]
        save_img = cv2.resize(save_img, (save_img.shape[1]*4,save_img.shape[0]*4))
        if len(face_recognition.face_encodings(save_img))==0:
            print("fail recognize new user")
            return "detecting"
        self.known_face_names.append(folder_name)
        face_encoding = face_recognition.face_encodings(save_img)[0]
        self.known_face_encodings.append(face_encoding)   
        self.conter[folder_name]=[-11,0]
        self.add_img(self.mkdir())
        return folder_name

    def add_img(self, name):
        self.conter[name][1] = time.time()
        if self.conter[name][1]-self.conter[name][0] < 1:  #save each 5 second
            return "save delay"
        self.conter[name][0]= time.time()
        folder = os.listdir(self.dirname+"/"+name)
        file_list = []  #not null
        for file in folder:
            if file == "info.txt":
                continue
            file_num = int(file.split("_")[1].split(".")[0])
            file_list.append(file_num)
        if len(file_list)==0:
            file_list=[0]
        if len(folder)>100:     #remove old img
            old_img = self.dirname+"/"+name+'/'+name +  '_'  + str(min(file_list))+".jpg"
            os.remove(old_img)
        file_name = self.dirname+"/"+name+'/'+name +  '_'  + str(max(file_list)+1)+".jpg"
        rgb_small_frame = self.frame[:, :, ::-1]
        save_img = rgb_small_frame[self.face_locations[0][0] : self.face_locations[0][2],self.face_locations[0][3]:self.face_locations[0][1]]
        save_img = cv2.resize(save_img, (save_img.shape[1],save_img.shape[0]))
        if len(face_recognition.face_encodings(save_img))==0:
            print("fail recognize img")
            return 
        cv2.imwrite(file_name,save_img)
        self.known_face_names.append(name +  '_'  + str(max(file_list)+1)+".jpg")
        face_encoding = face_recognition.face_encodings(save_img)[0]
        self.known_face_encodings.append(face_encoding) 
        print("save img")  
        return name +  '_'  + str(max(file_list)+1)+".jpg"

    def mkdir(self):
        folder = os.listdir(self.dirname)
        try :
            folder_name = str(max(map(int, folder))+1)
        except:
            folder_name = "1"
        os.makedirs(self.dirname+'/'+folder_name)
        print("new folder !!", folder_name)
        return folder_name

    def get_frame(self, location):
        # Grab a single frame of video
        rgb_small_frame = self.frame
        folder = os.listdir(self.dirname)
        rgb_small_frame  = rgb_small_frame[:, :, ::-1]

        # Resize frame of video to 1/4 size for faster face recognition processing
        # Only process every other frame of ivdeo to save time
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = location
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
        self.face_names = ""
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            predict_list = np.where(distances>0)
            predict_label = {}
            result =""
            min=1
            for i in predict_list[0]:
                if not self.known_face_names[i].split("_")[0] in predict_label.keys():
                    predict_label[self.known_face_names[i].split("_")[0]]=[1,distances[i]]
                else :
                    predict_label[self.known_face_names[i].split("_")[0]][0]+=1
                    predict_label[self.known_face_names[i].split("_")[0]][1]+=distances[i]

            for i in predict_label.keys():
                predict_label[i] =round(predict_label[i][1]/predict_label[i][0],3)

            for i in predict_label.keys():
                if min > predict_label[i]:
                    min = predict_label[i]
                    result = i

            if min > 0.45:
                name = self.new_user()
                print(predict_label)
            elif min <0.35 :
                name = result
                self.add_img(name)
            else :
                name = result
            print(name,min)
            self.face_names=name
        return str(self.face_names)

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
