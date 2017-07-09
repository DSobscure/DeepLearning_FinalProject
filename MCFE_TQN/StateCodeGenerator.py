import tensorflow as tf

class SCG():
    def __init__(self, code_size, feature_level):
        self.code_size = code_size
        self.feature_level = feature_level
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.random_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='random_code')

        self.state_code = []
        self.optimize = []
        for i in range(self.feature_level):
            state_code = self.build_network(self.state, trainable=True)      
            self.state_code.append(state_code)

            code_loss = tf.reduce_mean(tf.pow(state_code - self.random_code, 2))
            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(code_loss))          

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,3,3,1], padding='SAME') + conv1_bias        
        conv1_hidden = tf.nn.elu(conv1_hidden_sum)

        conv1_hidden_pool = tf.nn.max_pool(conv1_hidden, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden_pool, conv2_weight, strides = [1,1,1,1], padding='SAME') + conv2_bias
        conv2_hidden = tf.nn.elu(conv2_hidden_sum)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden = tf.nn.elu(conv3_hidden_sum)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 14*14*64])

        fc1_weight = tf.Variable(tf.truncated_normal([14*14*64, 1024], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [1024]), trainable = trainable)       
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden = tf.nn.elu(fc1_hidden_sum)

        fc2_weight = tf.Variable(tf.truncated_normal([1024, self.code_size], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)       
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden = tf.nn.sigmoid(fc2_hidden_sum)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        return fc2_hidden

    def update_code(self, sess, state, random_code):
        sess.run(self.optimize, feed_dict={self.state: state, self.random_code: random_code})

    def get_code(self, state):
        state_code = []
        for i in range(self.feature_level):
            state_code.append(self.state_code[i].eval(feed_dict={self.state: state}))

        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.feature_level):
                for k in range(self.code_size):
                    number *= 2
                    if state_code[j][i][k] > 0.5:
                        number += 1
            result.append(number)
        return result


