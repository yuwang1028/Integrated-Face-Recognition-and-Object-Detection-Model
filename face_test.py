"""---------------------------------------------------------
4. face recognition
1, open the camera, get the picture and grayscale
2. Face detection
3. Load the convolutional neural network model
4. Face recognition
------------------------------------------------------------"""
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

"""define parameters"""
faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 128          # take 100 pics
learning_rate = 0.01        # learning rate
size = 64                 # 64*64*3
imgs = []                 # save pics
labs = []                 # save labels
x = tf.placeholder(tf.float32, [None, size, size, 3])  # 64*64*3
y_ = tf.placeholder(tf.float32, [None, 2])
keep_prob_fifty = tf.placeholder(tf.float32)
keep_prob_seventy_five = tf.placeholder(tf.float32)


"""Define a function to read face data"""


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top,bottom,left,right = getPaddingSize(img)
            """放大图片扩充图片边缘部分"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)

def getPaddingSize(img):
    height, width, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(height, width)

    if width < longest:
        tmp = longest - width
        left = tmp // 2
        right = tmp - left
    elif height < longest:
        tmp = longest - height
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

"""Definition of neural network layer, a total of five layers, convolutional layer feature extraction, pooling layer dimension reduction, the full connection layer for classification, a total of two categories: my face (true), not my face (false)"""


def cnnLayer():
    """The first and second layers, input picture 64*64*3, output picture 32*32*32"""
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))                 # Convolution kernel size (3,3), input channel (3), output channel (32)
    b1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')+b1)    # 64*64*32, convolutional extraction features, increase the number of channels
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32*32*32, pooling dimension reduction, reduce complexity
    drop1 = tf.nn.dropout(pool1, keep_prob_fifty)

    """The third and fourth layers, input picture 32*32*32, output picture 16*16*64"""
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.conv2d(drop1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)        # 32*32*64
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 16*16*64
    drop2 = tf.nn.dropout(pool2, keep_prob_fifty)

    """The fifth and sixth layers, input picture 16*16*64, output picture 8*8*64"""
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
    b3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.conv2d(drop2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)        # 16*16*64
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       # 8*8*64=4096
    drop3 = tf.nn.dropout(pool3, keep_prob_fifty)

    """The seventh layer, the fully connected layer, input picture 8*8*64, reshape to 1*4096, and output 1*512"""
    Wf = tf.Variable(tf.random_normal([8*8*64,512]))     #
    bf = tf.Variable(tf.random_normal([512]))
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])         # ，1*4096
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)   # [1,4096]*[4096,512]=[1,512]
    dropf = tf.nn.dropout(dense, keep_prob_seventy_five)

    """Eighth layer, output layer, input 1*512, output 1*2, and then add, output a number"""
    Wout = tf.Variable(tf.random_normal([512,2]))
    bout = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dropf, Wout), bout)     # (1,512)*(512,2)=(1,2)
    return out

"""define face recognise function"""
def face_recognise(image):
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_fifty: 1.0, keep_prob_seventy_five: 1.0})
    if res[0] == 1:
        return "Yes,my face"
    else:
        return "No,other face"

if __name__ == '__main__':

    """1、read face data """
    readData(faces_my_path)
    readData(faces_other_path)
    imgs = np.array(imgs)  # 将图片数据与标签转换成数组
    labs = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs])
    """2、divide training and testing dataset"""
    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs, labs, test_size=0.05,
                                                            random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)
    train_x = train_x_2.astype('float32') / 255.0
    test_x = test_x_2.astype('float32') / 255.0
    print('Train Size:%s, Test Size:%s' % (len(train_x), len(test_x)))
    num_batch = len(train_x) // batch_size                    # 22784//128=178
    """3、Output the read face picture to the neural network and output out(1,2)"""
    out = cnnLayer()
    """4、Prediction: 1 indicates that the index of the maximum value in out is returned by row, rather than which one is larger than which one is returned. predict is the index value, 0 or 1, because the shape of out is (1,2), one row, two columns, and two numbers"""
    predict = tf.argmax(out, 1)   # 0 or 1

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    """5、Face detection, feature extractor: dlib comes with frontal_face_detector"""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)                  # open camera
    while True:
        _, img = cap.read()                    # load
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # graying
        dets = detector(gray_image, 1)
        if not len(dets):

            key = cv2.waitKey(30)
            if key == 27:
                sys.exit(0)
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
            """face 64*64"""
            face = img[x1:y1, x2:y2]
            face = cv2.resize(face, (size, size))
            """6、result"""
            print('It recognizes my face? %s' % face_recognise(face))

            cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
            if face_recognise(face) == "Yes,my face":
                cv2.putText(img, 'Yes,my face', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No,other face', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            """Draw rectangles by determining diagonal lines"""
            #cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('image', img)
        key = cv2.waitKey(30)
        if key == 27:
            sys.exit(0)

    sess.close()