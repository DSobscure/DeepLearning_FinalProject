import tensorflow as tf

class SCG():
    def __init__(self, code_size):
        self.code_size = code_size

        self.pre2_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='pre2_state')
        self.pre1_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='pre1_state')
        self.current_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='current_state')

        self.diff2_code = self.build_network(self.pre1_state - self.pre2_state, trainable=True)   
        self.diff1_code = self.build_network(self.current_state - self.pre1_state, trainable=True)   
        self.current_code = self.build_network(self.current_state, trainable=True)            

        self.diff2_code_loss = tf.reduce_mean(tf.pow(self.diff2_code - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2))
        self.diff1_code_loss = tf.reduce_mean(tf.pow(self.diff1_code - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2))
        self.current_code_loss = tf.reduce_mean(tf.pow(self.current_code - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2))

        self.optimize = []
        self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.diff2_code_loss))
        self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.diff1_code_loss))
        self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.current_code_loss))

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias            
        conv1_hidden = tf.nn.elu(conv1_hidden_sum)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden = tf.nn.elu(conv2_hidden_sum)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden = tf.nn.elu(conv3_hidden_sum)

        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*32])
        fc1_weight = tf.Variable(tf.truncated_normal([11*11*32, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)      
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden = tf.nn.elu(fc1_hidden_sum)

        fc2_weight = tf.Variable(tf.truncated_normal([512, self.code_size], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)      
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden = tf.nn.tanh(fc2_hidden_sum)

        code_layer = fc2_hidden
        print("code layer shape : %s" % code_layer.get_shape())

        return code_layer

    def update_code(self, sess, state):
        pre2_state = [x[0] for x in state]
        pre1_state = [x[1] for x in state]
        current_state = [x[2] for x in state]
        sess.run(self.optimize, feed_dict={self.pre2_state: pre2_state, self.pre1_state: pre1_state, self.current_state: current_state})

    def get_code(self, state):
        pre2_state = [x[0] for x in state]
        pre1_state = [x[1] for x in state]
        current_state = [x[2] for x in state]

        diff2_code = self.diff2_code.eval(feed_dict={self.pre2_state: pre2_state, self.pre1_state: pre1_state})
        diff1_code = self.diff1_code.eval(feed_dict={self.pre1_state: pre1_state, self.current_state: current_state})
        current_code = self.current_code.eval(feed_dict={self.current_state: current_state})
        
        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.code_size):
                number *= 2
                if diff2_code[i][j] > 0:
                    number += 1
            for j in range(self.code_size):
                number *= 2
                if diff1_code[i][j] > 0:
                    number += 1
            for j in range(self.code_size):
                number *= 2
                if current_code[i][j] > 0:
                    number += 1
            result.append(number)
        return result


