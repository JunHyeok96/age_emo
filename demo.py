from datetime import datetime
import time
import os
import cv2
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import base64
import io
import redis
from PIL import ImageFont, ImageDraw, Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from face_recog import *

r2 = redis.StrictRedis( port=6380)

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,   
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main(sess,age,gender,train_mode,images_pl):
    args = get_args()   
    depth = args.depth
    k = args.width
    face_recog = FaceRecog()
    # for face detection
    detector = dlib.get_frontal_face_detector()  #dlib face detect
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)
    emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'   
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

    # load model and weights
    img_size = 160
    font = cv2.FONT_HERSHEY_DUPLEX
    # capture video
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    inum=8
    while True:
        ts = datetime.now().strftime('%H:%M:%S')
        _, img = cap.read() 
        start_time = time.time()
        #ret, img = cap.read()
        img = cv2.resize(img,(640,480)) ### 400/300
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_recog.frame = input_img.copy()
        #face_recog.frame  = cv2.resize(face_recog.frame , (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)
        time.sleep(0.1)
        # detect faces using dlib detector
        detected = detector(input_img, 0)
        faces = np.empty((len(detected), img_size, img_size, 3))
        prob_list = []
        label_list = []
        id_list = []
        if len(detected)>0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                roi = gray[y1:y2, x1:x2]
                if roi.shape[0]==0 or roi.shape[1]==0:
                    continue
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                location = [(long(d.top()),long(d.right()),long(d.bottom()),long(d.left()))]
                id_info = face_recog.get_frame(location)
                preds = emotion_classifier.predict(roi)[0]    #emotion predict
                emotion_probability = np.max(preds)          #predict value
                prob_list.append(emotion_probability)           
                label2 = EMOTIONS[preds.argmax()]            #predict label
                label_list.append(label2)
                id_list.append(id_info)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 0), cv2.FILLED)
                if len(id_info)>0:
                    cv2.putText(img, "id = " + id_info , (x1 +6 , y2 - 6) ,font, 1.0, (255, 255, 255), 2)
                else :
                    cv2.putText(img, "detecting..." , (x1 +6 , y2 - 6) ,font, 1.0, (255, 255, 255), 2)
                faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
                ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
                label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
                draw_label(img, (d.left(), d.top()), label +" "+ label2)
                #print(label)
        else:
            pass
        # draw results
        for i in range(len(label_list)):
            aaggee = int(ages[i])
            if genders[i] == 0:
                ggeenn = 'woman'
            else:
                ggeenn = 'man'
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r_imgByteArr = io.BytesIO()
            r_imgByteArr.flush()
            img2 = Image.fromarray(img,'RGB')   #array2Img
            img2.save(r_imgByteArr, format='JPEG') 
            r_img_read = r_imgByteArr.getvalue()
            r_imgByteArr.flush()
            r_image_64_encode = base64.b64encode(r_img_read) 
            r_b64_numpy_arr = np.array(r_image_64_encode)
            send_id = 'result_'+str(inum)+"_"+str(i+1)
            r2.hmset(send_id,{'label':label_list[i],
            'prob':str(prob_list[i]),
            'img':str(r_b64_numpy_arr),
            'timestamp':ts,
            'pop':len(label_list),
            'age' :aaggee, 
            'gender':ggeenn,
            'id' : id_list[i]})
            
	print(round(time.time()-start_time,2))
        cv2.imshow("result", img)  # if don't need debuging remove imshow
        key = cv2.waitKey(1)

        if key == 27:
            break

def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess,age,gender,train_mode,images_pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
    args = parser.parse_args()
    sess, age, gender, train_mode,images_pl = load_network(args.model_path)
    main(sess,age,gender,train_mode,images_pl)
