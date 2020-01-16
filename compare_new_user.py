# face_recog.py

import face_recognition
import cv2
import os
import numpy as np
import shutil

class Detect_fake_user():
    def __init__(self, new_user_id):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.known_face_encodings = []
        self.known_face_names = []
        self.new_user_id =new_user_id
        # Load sample pictures and learn how to recognize it.
        self.dirname_target =  os.listdir('/etc/knowns/')
        self.dirname = '/etc/knowns/'+new_user_id
        for folder_name in self.dirname_target:
            if folder_name == new_user_id:
                continue
            files = os.listdir('/etc/knowns/'+folder_name)
            for filename in files:
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    self.known_face_names.append(name)
                    pathname = os.path.join('/etc/knowns/'+folder_name, filename)
                    img = face_recognition.load_image_file(pathname)
                    if len(face_recognition.face_encodings(img))<1:
                        os.remove(pathname)
                        continue
                    face_encoding = face_recognition.face_encodings(img)[0]
                    self.known_face_encodings.append(face_encoding)
        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def move_file(self, src_folder_name, dst_folder_name, target):
        src_files = os.listdir(src_folder_name)
        dst_files = os.listdir(dst_folder_name)
        file_list = []  #not null
        for file in dst_files:
            if file == "info.txt":
                continue
            file_num = int(file.split("_")[1].split(".")[0])
            file_list.append(file_num)
        if len(dst_files)>100:     #remove old img
            old_img = dst_folder_name+'/'+target +  '_'  + str(min(file_list))+".jpg"
            os.remove(old_img)
        start_file_name = max(file_list)+1
        for i in src_files:
            file_name = dst_folder_name+'/'+target +  '_'  + str(start_file_name)+".jpg"
            os.rename(src_folder_name+'/'+str(i), src_folder_name+'/'+target+"_"+str(start_file_name)+".jpg")
            shutil.move(src_folder_name+'/'+target+"_"+str(start_file_name)+".jpg", file_name)
            start_file_name+=1
        shutil.rmtree(src_folder_name)




    def compare_new_user(self):
        # Grab a single frame of video
        folder_name = self.dirname 
        files = os.listdir(folder_name)
        precision={}
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                pathname = os.path.join(folder_name, filename)
                frame = cv2.imread(pathname)
                rgb_small_frame = frame[:, :, ::-1]
                # Only process every other frame of video to save time
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
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
                        precision[i]=predict_label[i]
                        if min > predict_label[i]:
                            min = predict_label[i]
                        result = i
        min=1
        for i in precision.keys():
            if min > precision[i]:
                min = precision[i]
                result = i 
        if min <0.45:
            self.move_file(self.dirname, '/etc/knowns/'+ result, result)
            print("move_!! ",self.new_user_id, result, min)
        else:
            print("new user!!",self.new_user_id, result, min)



if __name__ == '__main__':
    new_user = "3"
    detect_fake_user = Detect_fake_user(new_user)
    detect_fake_user.compare_new_user()
    print('finish')