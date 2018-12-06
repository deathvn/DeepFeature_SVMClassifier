import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from glob import glob
import os, sys
import pickle
import vgg16
import utils
import cv2

filename = 'finalized_model.sav'

def train(trainFeat, testFeat, trainLabel, testLabel):
    print ("Training...")

    #train data
    clf = svm.SVC(gamma = 'scale')
    clf.fit(trainFeat, trainLabel)

    #Save model to disk
    pickle.dump(clf, open(filename, 'wb'))

    print(clf.n_support_)

def test(trainFeat, testFeat, trainLabel, testLabel):
    print ("Testing...")
    # load the model from disk
    clf = pickle.load(open(filename, 'rb'))

    print(clf.score(testFeat, testLabel))

    print(clf.n_support_)


def transform(image):
    cropped_image = cv2.resize(image, (224,224))
    return np.array(cropped_image)

def get_image(image_path):
    return transform(cv2.imread(image_path))

def extract_feature():
    # VGG16 fc6 cho bear label
    data = glob(os.path.join("./IMG_bear_vs_nonebear/bear/", "*.jpg"))
    batch = np.array([utils.load_image(img).reshape((224, 224, 3)) for img in data])

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [len(data), 224, 224, 3])
            feed_dict = {images: batch}

            print('Loading model...')
            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            print("Extracting feature...")
            fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)

            i=0
            for vect in fc6:
                np.save(data[i] + '.npy', vect)
                print(vect)
                i = i+1
            #print('FC6 feature: ', fc6)
            #print('Number of input: ', len(fc6))
            #print('Feature length of FC6: ', len(fc6[0]))

    # VGG16 fc6 cho nonebear label
    data = glob(os.path.join("./IMG_bear_vs_nonebear/nonebear/", "*.jpg"))
    batch = np.array([utils.load_image(img).reshape((224, 224, 3)) for img in data])

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [len(data), 224, 224, 3])
            feed_dict = {images: batch}

            print('Loading model...')
            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)
            print("Extracting feature...")
            fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)

            i=0
            for vect in fc6:
                np.save(data[i] + '.npy', vect)
                print(vect)
                i = i+1
            #print('FC6 feature: ', fc6)
            #print('Number of input: ', len(fc6))
            #print('Feature length of FC6: ', len(fc6[0]))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py [train, test]")
    else:
        cmd = sys.argv[1]

        #print ("Training...")
        #clf = svm.SVC(gamma = 'scale')
        #clf.fit(trainFeat, trainLabel)
        #print ("Testing...")
        #print(clf.score(testFeat, testLabel))
        #print(clf.n_support_)

        if cmd == "feature":
            extract_feature()
        else:

            X = []
            y = []
            #bear vector (1)
            path1 = "./vgg16/bear/"
            folder = os.listdir(path1)
            for fol in folder:
                s = path1 + fol
                #print (s)
                y.append(1)
                X.append(np.load(s).flatten())

            #nonebear vector (0)
            path2 = "./vgg16/nonebear/"
            folder = os.listdir(path2)
            for fol in folder:
                s = path2 + fol
                #print(s)
                y.append(0)
                X.append(np.load(s).flatten())

            y = np.array(y)
            X = np.array(X)

            #20 pic for train and 10 pic for test
            (trainFeat, testFeat, trainLabel, testLabel) = train_test_split(X, y, test_size=(1/3), random_state=42)

            if cmd == "train":
                train(trainFeat, testFeat, trainLabel, testLabel)
            elif cmd == "test":
                test(trainFeat, testFeat, trainLabel, testLabel)
            else:
                print("Usage: python main.py [train, test]")

