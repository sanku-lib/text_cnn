'''
Module: Text CNN
Author: Shibsankar Das
'''


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import time
import os


def preprocess_data(data_src, x_col_name, y_col_name):
    data = pd.read_csv(data_src,encoding='ISO-8859-1')
    data.dropna(how='any',inplace=True)
    input_X = data[x_col_name]
    input_Y = data[y_col_name]
    labels = list(set(input_Y))
    one_hot_matrix = np.zeros([len(labels),len(labels)])
    np.fill_diagonal(one_hot_matrix,1)
    label_dict = dict(zip(labels,one_hot_matrix))
    max_document_length = max(list(len(x.split(' ')) for x in input_X))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X = np.asarray(list(vocab_processor.fit_transform(input_X)))
    Y = np.asarray(input_Y.apply(lambda y: label_dict[y]))
    return X,Y,labels,vocab_processor


data_src = './data_repository/INPUT_DATASET.csv'
X_col_name = 'X_SOURCE_COL_NAME'
Y_col_name = 'Y_TARGET_COL_NAME'
X,Y,labels,vocab_processor = preprocess_data(data_src,X_col_name,Y_col_name)


# Parameters
num_epochs = 10
sequence_length = X.shape[1]
emb_dimension = 128
n_classes = len(labels)
vocab_size = len(vocab_processor.vocabulary_)
filter_sizes = [3,4,5]
num_filters = 128
dropout_keep_probs = 0.5
l2_reg_lambda = 0
batch_size = 128
evaluate_every = 100


# Flow Diagram of CNN
input_x = tf.placeholder(dtype=tf.int32,shape=[None,sequence_length],name='input_x')
input_y = tf.placeholder(dtype=tf.int32,shape=[None,n_classes],name='input_y')
dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

# Embeddings Layer
with tf.name_scope('embeddings'):
    W = tf.Variable(tf.random_uniform([vocab_size,emb_dimension],-1.0,1.0),name='W')
    embedded_chars = tf.nn.embedding_lookup(W,input_x)
    embedded_chars_extended = tf.expand_dims(embedded_chars,-1)

# Convolution Layer
pooled_output = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope('conv-maxpool-%s' % i):
        filter_shape = [filter_size,emb_dimension,1,num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape,stddev=1.0), name='W')
        b = tf.Variable(tf.constant(0.1,shape=[num_filters]), name='b')
        conv = tf.nn.conv2d(
            embedded_chars_extended,
            W,
            strides=[1,1,1,1],
            padding='VALID',
            name = 'conv_2d'
        )

        conv_relu = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
        conv_relu_maxpool = tf.nn.max_pool(
            conv_relu,
            ksize = [1,sequence_length-filter_size+1,1,1],
            strides=[1,1,1,1],
            padding='VALID',
            name = 'pool'
        )
        pooled_output.append(conv_relu_maxpool)
    

# Fully connected Layer
number_of_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_output,3)
h_pool_flat = tf.reshape(h_pool,[-1,number_of_filters_total])

# Add dropout
with tf.name_scope('dropout'):
    h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_probs)


# Final (unnormalized) scores and predictions
# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)
with tf.name_scope('output'):
    W = tf.get_variable(
        'W',
        shape=[number_of_filters_total, n_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
    probabilities = tf.nn.softmax(scores, name='probabilities')
    predictions = tf.argmax(scores, 1, name='predictions')

# Calculate mean cross-entropy loss
with tf.name_scope('loss'):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels = input_y, logits = scores) #  only named arguments accepted            
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# Accuracy
with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

with tf.name_scope('num_correct'):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')

"""Step 2: split the original dataset into train and test sets"""
x_, x_test, y_, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
"""Step 3: shuffle the train set and split the train set into train and dev sets"""
# shuffle_indices = np.random.permutation(np.arange(len(y_)))
shuffle_indices = np.random.permutation(np.arange(len(y_)))
x_shuffled = x_[shuffle_indices]
y_shuffled = y_[shuffle_indices]
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

"""Step 5: build a graph and cnn object"""
graph = tf.Graph()
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver()

# One training step: train the model with one batch
def train_step(x_batch, y_batch):
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
        dropout_keep_prob: dropout_keep_probs}
    _, step, losses, acc = sess.run([train_op, global_step, loss, accuracy], feed_dict)
    print(step,' : ',losses,' : ',acc)


# One evaluation step: evaluate the model with one batch
def dev_step(x_batch, y_batch):
    feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0}
    step, losses, acc, num_corrects = sess.run([global_step, loss, accuracy, num_correct], feed_dict)
    return num_corrects


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# Save the word_to_id map since predict.py needs it
vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
sess.run(tf.global_variables_initializer())

# Training starts here
train_batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
best_accuracy, best_at_step = 0, 0

"""Step 6: train the cnn model with x_train and y_train (batch by batch)"""
for train_batch in train_batches:
    x_train_batch, y_train_batch = zip(*train_batch)
    train_step(x_train_batch, y_train_batch)
    current_step = tf.train.global_step(sess, global_step)

    """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
    if current_step % evaluate_every == 0:
        dev_batches = batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
        total_dev_correct = 0
        for dev_batch in dev_batches:
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
            total_dev_correct += num_dev_correct

        dev_accuracy = float(total_dev_correct) / len(y_dev)
        print('Accuracy on dev set: {}'.format(dev_accuracy))

        """Step 6.2: save the model if it is the best based on accuracy on dev set"""
        if dev_accuracy >= best_accuracy:
            best_accuracy, best_at_step = dev_accuracy, current_step
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print('Saved model at {} at step {}'.format(path, best_at_step))
            print('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))
            
            
"""Step 7: predict x_test (batch by batch)"""
test_batches = batch_iter(list(zip(x_test, y_test)), batch_size, 1)
total_test_correct = 0
for test_batch in test_batches:
    x_test_batch, y_test_batch = zip(*test_batch)
    num_test_correct = dev_step(x_test_batch, y_test_batch)
    total_test_correct += num_test_correct

test_accuracy = float(total_test_correct) / len(y_test)
print('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
print('The training is complete')

