"""--------------------------------------------------------------
3. CNN model training
Training model: A total of eight layers of neural network, convolutional layer feature extraction, pooling layer dimension reduction, fully connected layer classification.
Training data: 22784, test data: 1200, training set: test set =20:1
There are two categories: My face (yes) and not my face (no).
A total of eight layers: the first and second layers (convolution layer 1, pooling layer 1), input picture 64*64*3, output picture 32*32*32
The third and fourth layers (convolution layer 2, pooling layer 2), input picture 32*32*32, output picture 16*16*64
The fifth and sixth layers (convolution layer 3, pooling layer 3), input picture 16*16*64, output picture 8*8*64
The seventh layer (full connection layer), input picture 8*8*64, reshape to 1*4096, output 1*512
The eighth layer (output layer), input 1*512, output 1*2
Learning rate: 0.01
Loss function: cross entropy
Optimizer: Adam
------------------------------------------------------------------"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import cv2
import sys
import os

"""define parameters"""
faces_my_path = './faces_my'
faces_other_path = './faces_other'
batch_size = 100
learning_rate = 0.01
size = 64
labs = []
imgs = []

"""Define a function to read face data and assign different onehot values according to different names"""
def readData(path , h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            """Enlarge the image to expand the edges of the image"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)

"""Define the dimensional transformation function with a set of formulas"""


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
    drop1 = tf.nn.dropout(pool1, keep_prob_fifty)      # Randomly discard some neurons with a certain probability to achieve higher training speed and prevent overfitting

    """The third and fourth layers, input picture 32*32*32, output picture 16*16*64"""
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # Convolution kernel size (3,3), input channel (32), output channel (64)
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


    """The seventh layer, the fully connected layer, flattens the convolutional output of the picture into a one-dimensional vector, the input picture 8*8*64, reshape to 1*4096, and the output 1*512"""
    Wf = tf.Variable(tf.random_normal([8*8*64,512]))     # input channel (4096)， output channel (512)
    bf = tf.Variable(tf.random_normal([512]))
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)   # [1,4096]*[4096,512]=[1,512]
    dropf = tf.nn.dropout(dense, keep_prob_seventy_five)

    """Eighth layer, output layer, input 1*512, output 1*2, and then add, output a number"""
    Wout = tf.Variable(tf.random_normal([512,2]))        # input channel(512)， output channel(2)
    bout = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dropf, Wout), bout)     # (1,512)*(512,2)=(1,2)
    return out

"""def traninng function"""
def train():
    out = cnnLayer()
    """The loss function is cross entropy"""
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    """Adam optimizer is used"""
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    """Find the accuracy, compare the labels to see if they are equal, and average all the numbers."""
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
        for n in range(10):

             #"""num_batch = len(train_x) // batch_size """
            for i in range(num_batch):
                batch_x = train_x[i*batch_size: (i+1)*batch_size]          # 图片
                batch_y = train_y[i*batch_size: (i+1)*batch_size]          # 标签：[0,1] [1,0]
                _, loss, summary = sess.run([optimizer, cross_entropy, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y,
                                                       keep_prob_fifty: 0.5, keep_prob_seventy_five: 0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                print("step:%d,  loss:%g" % (n*num_batch+i, loss))

                if (n*num_batch+i) % 100 == 0:
                    acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_fifty: 1.0, keep_prob_seventy_five: 1.0})
                    print("step:%d,  acc:%g" % (n*num_batch+i, acc))
                    """stop training until acc>98%"""
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)


if __name__ == '__main__':

    """1、read face dataset"""
    readData(faces_my_path)
    readData(faces_other_path)
    imgs = np.array(imgs)                   # Converts image data and labels into arrays
    labs = np.array([[0, 1] if lab == faces_my_path else [1, 0] for lab in labs])  # [0,1]is my face，[1,0]is other's face
    """2、Randomly divide test set and training set"""
    train_x_1, test_x_1, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
    train_x_2 = train_x_1.reshape(train_x_1.shape[0], size, size, 3)        #
    test_x_2 = test_x_1.reshape(test_x_1.shape[0], size, size, 3)
    """3、normalization"""
    train_x = train_x_2.astype('float32')/255.0
    test_x = test_x_2.astype('float32')/255.0
    print('Train Size:%s, Test Size:%s' % (len(train_x), len(test_x)))

    num_batch = len(train_x) // batch_size
    x = tf.placeholder(tf.float32, [None, size, size, 3])                 # 输入X：64*64*3
    y_ = tf.placeholder(tf.float32, [None, 2])                            # 输出Y_：1*2
    keep_prob_fifty = tf.placeholder(tf.float32)                          # 50%，即0.5
    keep_prob_seventy_five = tf.placeholder(tf.float32)                   # 75%，即0.75
    """4、train"""
    train()