import os
import inception_resnet_v1
import dlib
import numpy as np
import argparse
import cv2
import tensorflow as tf
from imutils.face_utils import FaceAligner
import face_recognition


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


def main(sess,age,gender,train_mode,images_pl):
    args = get_args()   
    depth = args.depth
    k = args.width
    # for face detection
    img_size = 160
    detector = dlib.get_frontal_face_detector()  #dlib face detect
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)
    dirname = '/etc/knowns'
    files_folder = os.listdir(dirname)
    for folder in files_folder:
        folder_data = os.listdir(dirname+"/"+folder)
        age_ave=0
        age_cnt=0
        for filename in folder_data:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                pathname = os.path.join(dirname+"/"+folder, filename)
                img = face_recognition.load_image_file(pathname)
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_h, img_w, _ = np.shape(input_img)
                detected = detector(input_img, 0)
                faces = np.empty((len(detected), img_size, img_size, 3))
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
                if len(detected) > 0:
                    # predict ages and genders of the detected faces
                    ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
                    age_ave+=ages
                    age_cnt+=1
        f = open(dirname+ '/'+folder+'/'+ "age.txt", "w")
        f.write(str(int(age_ave/age_cnt)))
        f.close()        
        print(folder+" is done")



    # load model and weights




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
