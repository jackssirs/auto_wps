import tensorflow as tf
import collections
import fcntl
import cv2
import os
import numpy as np
import time
from numpy import *
import socket

host=''
ompileport=1234
wpshost='192.168.7.151'
wpsport=1235



# Parameters
learning_rate = 1e-3
training_epochs = 15

trainbatch_size = 50

wsize = 640
hsize = 480
train_nums = 912
test_nums = 116
display_step = 1

Train_all = "data/train/"
Test_all = "data/test/"
print("Initializing...")

train_images_names = [0 for i in range(train_nums)]
train_images = array([[[0 for j in range(wsize)] for k in range(hsize)] for l in range(trainbatch_size)])
test_images = array([[[0 for j in range(wsize)] for k in range(hsize)] for l in range(test_nums)])
pred_images = array([[[0 for j in range(wsize)] for k in range(hsize)] for l in range(1)])

#save training image name in shuffle(random) order
i = 0
for img in os.listdir(Train_all):
  train_images_names[i] = img
  i = i + 1
random.shuffle(train_images_names)

#labels: bad=0,1 good=1,0
train_labels = array([[0 for i in range(5)] for k in range(trainbatch_size)])
test_labels = array([[0 for i in range(5)] for k in range(test_nums)])

Datasets = collections.namedtuple('Datasets', ['train_images', 'train_labels'])
DatasetsTest = collections.namedtuple('DatasetsTest', ['test_images', 'test_labels'])

current = 0
def get_next_trainbatch():
  global trainbatch_size
  global train_nums
  global current
  batch_realsize = trainbatch_size
  if (train_nums - current) <= trainbatch_size:
    batch_realsize = (train_nums - current)
  for i in range(batch_realsize):
    img = train_images_names[current+i]
    #print(img)
    train_images[i] = cv2.imread(Train_all+img,cv2.IMREAD_GRAYSCALE)
    flag = img[0]
    if flag == "a" :
      train_labels[i] = array([1,0,0,0,0])
    elif flag == "b":
      train_labels[i] = array([0,1,0,0,0])
    elif flag == "c":
      train_labels[i] = array([0,0,1,0,0])
    elif flag == "d":
      train_labels[i] = array([0,0,0,1,0])
    elif flag == "e":
      train_labels[i] = array([0,0,0,0,1])
    #print(train_labels[i]);

  current = current + batch_realsize

  return Datasets(train_images=train_images, train_labels=train_labels)

def get_testimages():
  global test_nums
  i = 0
  for img in os.listdir(Test_all):
    test_images[i] = cv2.imread(Test_all+img,cv2.IMREAD_GRAYSCALE)
    flag = img[0]
    if flag == "a" :
      test_labels[i] = array([1,0,0,0,0])
    elif flag == "b":
      test_labels[i] = array([0,1,0,0,0])
    elif flag == "c":
      test_labels[i] = array([0,0,1,0,0])
    elif flag == "d":
      test_labels[i] = array([0,0,0,1,0])
    elif flag == "e":
      test_labels[i] = array([0,0,0,0,1])
    i = i + 1

  return DatasetsTest(test_images=test_images, test_labels=test_labels)

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 256 # 2nd layer num features
n_input = hsize * wsize 
n_classes = 5

# tf Graph input
x = tf.placeholder("float", [None, hsize , wsize])
y = tf.placeholder("float", [None, n_classes])
tst = tf.placeholder(tf.bool)
x_image = tf.reshape(x,[-1,n_input])
keep_prob = tf.placeholder(tf.float32)

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with sigmoid activation
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with sigmoid activation
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])) #Hidden layer with RELU activation
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    layer_4 = tf.matmul(layer_3, _weights['out']) + _biases['out']
    return layer_4

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x_image, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    checkpoint = tf.train.get_checkpoint_state("model")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        img_cap = "./mem_image/cap.png"
        lock = "./mem_image/lock"  #add a lock file to sychronize with file img_cap (in cap_img.py)

        #touch lock, so cap_image can save
        os.system("touch "+lock)

        maybe_crash = 0
        while (True):
            #lock exists means we just predicted it and cap_image is saving newer image
            if os.path.exists(lock):
                continue;
            pred_image = cv2.imread(img_cap,cv2.IMREAD_GRAYSCALE)
            pred_images[0] = pred_image
            prediction = array([sess.run(pred, feed_dict={x: pred_images, keep_prob:1})[0]])
            pred_res = sess.run(tf.argmax(prediction,1))
            if(pred_res[0] == 0):
                print ("0: order step.")
            elif pred_res[0] == 1:
                print ("1: pre open wps.")
            elif pred_res[0] == 2:
                print ("2: wps opened.")
            elif pred_res[0] == 3:
                print ("3: wps shows end page of doc.")
            elif pred_res[0] == 4:
                maybe_crash = maybe_crash + 1
                print ("maybe_crash: " + str(maybe_crash))
                if 50 <= maybe_crash:
                    print ("4: wps crashed.")
            else:
                print ("bug")

            #touch lock, so cap_image can save
            os.system("touch "+lock)
                
        exit(0)
        
    else:
        print("Could not find old network weights")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_nums/trainbatch_size )
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = get_next_trainbatch()
            if i% 10 == 0:
                print ("Step ",i,", Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys, keep_prob: 1}))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    saver.save(sess,'./model/wps-deeptimer')
    print ("Optimization Finished!")


    # Test model
    batchtest = get_testimages()
    print ("Final Test Accuracy:", accuracy.eval({x: batchtest[0], y: batchtest[1], keep_prob: 1}))
