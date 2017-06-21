import tensorflow as tf

class SCG():
    def __init__(self, code_size, feature_count):
        self.code_size = code_size
        self.feature_count = feature_count
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.code = []
        for i in range(self.feature_count):
            self.code.append(self.build_network(self.state, trainable=True))
        self.loss = 0
        for i in range(self.feature_count):
            self.loss  += tf.reduce_mean(tf.pow(self.code[i] - tf.random_uniform(shape = [self.code_size]), 2))       
        self.optimize = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)   
    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias         
        conv1_hidden = tf.nn.elu(conv1_hidden_sum)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden = tf.nn.elu(conv2_hidden_sum)
        conv2_hidden_flat = tf.reshape(conv2_hidden, [-1, 11*11*32])

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*32, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)      
        fc1_hidden_sum = tf.matmul(conv2_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden = tf.nn.elu(fc1_hidden_sum)

        fc2_weight = tf.Variable(tf.truncated_normal([512, self.code_size], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)      
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden = tf.nn.sigmoid(fc2_hidden_sum)

        code_layer = fc2_hidden
        print("code layer shape : %s" % code_layer.get_shape())

        return code_layer

    def update_code(self, sess, state):
        sess.run([self.optimize], feed_dict={self.state : state})

    def get_code(self, state):
        code = []
        for i in range(self.feature_count):
            code.append(self.code[i].eval(feed_dict={self.state: state}))
        result = []
        for i in range(len(state)):
            number = 0
            for c in range(self.feature_count):
                for j in range(self.code_size):
                    number *= 2
                    if code[c][i][j] > 0.5:
                        number += 1
            result.append(number)
        return result


