# install the package
# pip install tensorflow
# pip install keras
# pip install scikit-learn
"""-----------------------------------------
1. First, collect my face data set
Get my face data set 10000, using dlib
Face recognition, although the speed is slower than OpenCV recognition, but the recognition effect
The fruit is better.
Face size: 64*64
-----------------------------------------"""
import cv2
import dlib
import os
import random

faces_my_path = './faces_my'
size = 64
if not os.path.exists(faces_my_path):
    os.makedirs(faces_my_path)

"""Change image parameters: brightness and contrast"""


def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,k] = tmp
    return img



"""Feature extractor: frontal face detecto comes with dlib"""
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

num = 1
while True:
    if (num <= 10000):
        print('Being processed picture %s' % num)
        success, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """use feature extractor to detect"""
        dets = detector(gray_img, 1)
        """--------------------------------------------------------------------
        The enumerate function is used to traverse the elements in the sequence and their subscripts. i is the face number and d is the element corresponding to i. 
        left: The distance between the left side of the face and the left edge of the picture; right: The distance between the right side of the face and the left edge of the image 
        top: the distance between the upper part of the face and the upper part of the image; bottom: The distance between the bottom of the face and the top border of the picture
        ----------------------------------------------------------------------"""
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            """Adjust the contrast and brightness of the picture. The contrast and brightness values are random numbers, which can increase the diversity of the sample"""
            face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size,size))
            cv2.imshow('image', face)
            cv2.imwrite(faces_my_path+'/'+str(num)+'.jpg', face)
            num += 1
        key = cv2.waitKey(30)
        if key == 27:
            break
    else:
        print('Finished!')
        break



# Using OpenCV and other tools for image processing, the face image is adjusted to the same size and grayscale.
# In addition, labels (identifiers corresponding to faces) are encoded for use in model training.


def prepare_data(data_folder_path):
    faces = []
    labels = []

    for folder_name in os.listdir(data_folder_path):
        label = int(folder_name)
        subject_path = os.path.join(data_folder_path, folder_name)

        for filename in os.listdir(subject_path):
            img_path = os.path.join(subject_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(np.asarray(img, dtype=np.uint8))
            labels.append(label)

    return faces, labels


# A convolutional neural network (CNN) model is constructed using Keras for face recognition. Here's a simple example:
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model and train it using the collected face data set:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(faces), np.array(labels), epochs=10)






# Write a test script, load the trained model and perform the face recognition test:
import cv2
import numpy as np


def recognize_face(model, face_img):

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (width, height))
    face_img = np.reshape(face_img, (1, width, height, 1))


    result = model.predict(face_img)
    label = np.argmax(result)

    return label


model = load_model('face_recognition_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face_label = recognize_face(model, frame)

    cv2.putText(frame, str(face_label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# save the model
model.save('face_recognition_model.h5')
