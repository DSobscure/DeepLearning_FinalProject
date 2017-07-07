import tensorflow as tf

class SCG():
    def __init__(self, code_size, feature_level):
        self.code_size = code_size
        self.feature_level = feature_level
        self.previous2_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='previous2_state')
        self.previous_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='previous_state')
        self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='state')
        self.random_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='random_code')

        self.state_code = []
        self.difference_code = []
        self.optimize = []
        for i in range(self.feature_level):
            state_code, state_code2 = self.build_network(self.state, self.previous_state, trainable=True)   
            difference_code, difference_code2 = self.build_network(self.state - self.previous_state, self.previous_state - self.previous2_state, trainable=True)          
            self.state_code.append(state_code)
            self.difference_code.append(difference_code)

            code_loss = tf.reduce_mean(tf.pow(state_code - self.random_code, 2)) + tf.reduce_mean(tf.pow(state_code - state_code2, 2))
            difference_code_loss = tf.reduce_mean(tf.pow(self.difference_code[i] - self.random_code, 2)) + tf.reduce_mean(tf.pow(difference_code - difference_code2, 2))

            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(code_loss))
            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(difference_code_loss))

    def build_network(self, x, x2, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,3,3,1], padding='SAME') + conv1_bias        
        conv1_hidden = tf.nn.elu(conv1_hidden_sum)
        conv1_hidden_sum2 = tf.nn.conv2d(x2, conv1_weight, strides = [1,3,3,1], padding='SAME') + conv1_bias        
        conv1_hidden2 = tf.nn.elu(conv1_hidden_sum2)

        conv1_hidden_pool = tf.nn.max_pool(conv1_hidden, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        conv1_hidden_pool2 = tf.nn.max_pool(conv1_hidden2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden_pool, conv2_weight, strides = [1,1,1,1], padding='SAME') + conv2_bias
        conv2_hidden = tf.nn.elu(conv2_hidden_sum)
        conv2_hidden_sum2 = tf.nn.conv2d(conv1_hidden_pool2, conv2_weight, strides = [1,1,1,1], padding='SAME') + conv2_bias
        conv2_hidden2 = tf.nn.elu(conv2_hidden_sum2)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden = tf.nn.elu(conv3_hidden_sum)
        conv3_hidden_sum2 = tf.nn.conv2d(conv2_hidden2, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden2 = tf.nn.elu(conv3_hidden_sum2)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 14*14*32])
        conv3_hidden_flat2 = tf.reshape(conv3_hidden2, [-1, 14*14*32])

        fc1_weight = tf.Variable(tf.truncated_normal([14*14*32, self.code_size], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)       
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden = tf.nn.sigmoid(fc1_hidden_sum)
        fc1_hidden_sum2 = tf.matmul(conv3_hidden_flat2, fc1_weight) + fc1_bias
        fc1_hidden2 = tf.nn.sigmoid(fc1_hidden_sum2)

        print("code layer shape : %s" % fc1_hidden.get_shape())

        return fc1_hidden, fc1_hidden2

    def update_code(self, sess, state, random_code):
        state0 = [s[0] for s in state]
        state1 = [s[1] for s in state]
        state2 = [s[2] for s in state]
        sess.run(self.optimize, feed_dict={self.previous2_state : state0, self.previous_state : state1, self.state: state2, self.random_code: random_code})

    def get_code(self, state):
        state0 = [s[0] for s in state]
        state1 = [s[1] for s in state]
        state2 = [s[2] for s in state]

        state_code = []
        difference_code = []
        for i in range(self.feature_level):
            state_code.append(self.state_code[i].eval(feed_dict={self.state: state2}))
            difference_code.append(self.difference_code[i].eval(feed_dict={self.previous_state: state1, self.state: state2}))

        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.feature_level):
                for k in range(self.code_size):
                    number *= 2
                    if state_code[j][i][k] > 0.5:
                        number += 1
                for k in range(self.code_size):
                    number *= 2
                    if difference_code[j][i][k] > 0.5:
                        number += 1
            result.append(number)
        return result


