
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df = df.drop(['Name', 'Embarked', 'Ticket'], 1)
df.fillna(0, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

all_cabins = df['Cabin'].tolist()
all_cabins_set = set(all_cabins)
cabins_map = {}
x = 0
for cabin in all_cabins_set:
    cabins_map[cabin] = x
    x += 1
    
df['Cabin'] = df['Cabin'].map(cabins_map)
df = df.astype('float32')

labels = df['Survived']
labels = (np.arange(2) == (df['Survived'])[:,None]).astype(np.float32)
df = df.drop(['Survived'], 1)
train_data = np.array(df[:800])
test_data = np.array(df[800:])
train_labels = labels[:800]
test_labels = labels[800:]


# In[4]:

import tensorflow as tf

num_classes = 2
data_size = 8
learning_rate = 0.002
epochs = 15000

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def nn_model(data, weights, biases):
    layer_1 = tf.matmul(data, weights['lay1']) + biases['lay1']
    relu = tf.nn.relu(layer_1)
    relu_d = tf.nn.dropout(relu, pkeep)
    layer_2 = tf.matmul(relu_d, weights['lay2']) + biases['lay2']
    return layer_2

def accuracy(predicitions, labels):
    return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

graph = tf.Graph()

with graph.as_default():
    
    with tf.name_scope('Input'):
        tf_train_data = tf.placeholder(tf.float32, shape=(None, data_size), name='passengers')
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        tf_test_data = tf.constant(test_data, name='test_passengers')
        pkeep = tf.placeholder(tf.float32, name="pkeep")
    
    with tf.name_scope('Layers'):
        weights = {
            'lay1': tf.Variable(tf.truncated_normal([data_size, 100], stddev=0.1), name='weights1'),
            'lay2': tf.Variable(tf.truncated_normal([100, num_classes], stddev=0.1), name='weights2')
        }

        biases = {
            #Important to not define initial biases being passed to relu as zeros to prevent the "dying relu" problem 
            'lay1': tf.Variable((tf.ones([100])/10), name='biases1'), 
            'lay2': tf.Variable(tf.zeros([num_classes]), name='biases2')
        }

        logits = nn_model(tf_train_data, weights, biases)
        
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_train_labels)) 
        tf.summary.scalar('loss', loss)
    
    with tf.name_scope('Optimizer'):
        #AdamOptimizer yields better results but GDOptimizer yields better visualization of weights, why?
        #Also need to learn more about how these optimizers work 
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) 
    
    with tf.name_scope('Evaluation'):
        train_prediction = tf.nn.softmax(logits) ### Think about renaming this 
        correct_prediction = tf.equal(tf.argmax(train_prediction,1), tf.argmax(tf_train_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         tf.summary.scalar('accuracy', accuracy)

with tf.Session(graph=graph) as sess:
    #For tensorboard
#     train_writer = tf.summary.FileWriter(os.path.join(DIR, 'train'), sess.graph)
#     test_writer = tf.summary.FileWriter(os.path.join(DIR, 'test'), sess.graph)
#     summary_op = tf.summary.merge_all()
    
    #Training model
    sess.run(tf.global_variables_initializer())
    print('Initialized variables.')
    with tf.name_scope('training'):
        for epoch in range(1,epochs):
            train_data, train_labels = randomize(train_data, train_labels)
            _ = sess.run(optimizer, feed_dict={tf_train_data: train_data, tf_train_labels: train_labels, pkeep: 0.70})
            
            if (epoch % 1000 == 0):
                print('Epoch: ' + str(epoch) + ' Accuracy: ' + str(sess.run(accuracy, feed_dict={tf_train_data: train_data, tf_train_labels: train_labels, pkeep: 1.0})) + ' Loss: ' + str(sess.run(loss, feed_dict={tf_train_data: train_data, tf_train_labels: train_labels, pkeep: 1.0})))
        
        #Need to figure out how to correctly display test data results in an informative way on tensorboard
        print('Test accuracy: ' + str(sess.run(accuracy, feed_dict={tf_train_data: test_data, tf_train_labels: test_labels, pkeep: 1.0})))
        


# In[ ]:



