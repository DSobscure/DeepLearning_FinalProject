import tensorflow as tf

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True)

class SCG():
    def __init__(self, code_size, code_level):
        self.code_size = code_size
        self.code_level = code_level
        
        self.previous_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='previous_state')
        self.current_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='current_state')
        
        self.original_current_states = [self.current_state]
        self.original_change_states = [self.current_state - self.previous_state]
        
        self.rec_current_states = []
        self.rec_change_states = []
        
        self.current_codes = []
        self.change_codes = []

        for i in range(self.code_level):
            rec_current_state, current_code = self.build_network(self.original_current_states[i], trainable=True)  
            rec_change_state, change_code = self.build_network(self.original_change_states[i], trainable=True)  
            
            self.rec_current_states.append(rec_current_state)
            self.rec_change_states.append(rec_change_state)

            self.current_codes.append(current_code)
            self.change_codes.append(change_code)

            self.original_current_states.append(self.original_current_states[i] - rec_current_state)
            self.original_change_states.append(self.original_change_states[i] - rec_change_state)

        self.rec_current_loss = []
        self.rec_change_loss = []
        self.current_code_loss = []
        self.change_code_loss = []

        for i in range(self.code_level):
            self.rec_current_loss.append(tf.reduce_mean(tf.pow(self.rec_current_states[i] - self.original_current_states[i], 2)))
            self.rec_change_loss.append(tf.reduce_mean(tf.pow(self.rec_change_states[i] - self.original_change_states[i], 2)))

            self.current_code_loss.append(tf.reduce_mean(tf.pow(self.current_codes[i] - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2)))
            self.change_code_loss.append(tf.reduce_mean(tf.pow(self.change_codes[i] - tf.random_uniform(shape = [self.code_size],minval=-1,maxval=1), 2)))

            print('build loss', i)

        self.optimize = []
        for i in range(self.code_level):
            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.rec_current_loss[i] + self.current_code_loss[i]))
            self.optimize.append(tf.train.RMSPropOptimizer(0.00025).minimize(self.rec_change_loss[i] + self.change_code_loss[i]))
            print('build optimize', i)

        self.total_rec_current_loss = sum(self.rec_current_loss)
        self.total_rec_change_loss = sum(self.rec_change_loss)
        self.total_current_code_loss = sum(self.current_code_loss)
        self.total_change_code_loss = sum(self.change_code_loss)

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)             
        conv1_hidden_sum = tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias      
        conv1_hidden_bn = batch_norm(conv1_hidden_sum)        
        conv1_hidden = tf.nn.elu(conv1_hidden_bn)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv2_hidden_sum = tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias
        conv2_hidden_bn = batch_norm(conv2_hidden_sum)
        conv2_hidden = tf.nn.elu(conv2_hidden_bn)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)       
        conv3_hidden_sum = tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias
        conv3_hidden_bn = batch_norm(conv3_hidden_sum)
        conv3_hidden = tf.nn.elu(conv3_hidden_bn)

        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*32])
        fc1_weight = tf.Variable(tf.truncated_normal([11*11*32, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)      
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden_bn = batch_norm(fc1_hidden_sum)
        fc1_hidden = tf.nn.elu(fc1_hidden_bn)

        fc2_weight = tf.Variable(tf.truncated_normal([512, self.code_size], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)      
        fc2_hidden_sum = tf.matmul(fc1_hidden, fc2_weight) + fc2_bias
        fc2_hidden_bn = batch_norm(fc2_hidden_sum)
        fc2_hidden = tf.nn.tanh(fc2_hidden_bn)

        code_layer = fc2_hidden
        print("code layer shape : %s" % code_layer.get_shape())

        dfc1_weight = tf.Variable(tf.truncated_normal([self.code_size, 512], stddev = 0.02))
        dfc1_bias = tf.Variable(tf.constant(0.02, shape = [512]))
        dfc1_hidden_sum = tf.matmul(fc2_hidden, dfc1_weight) + dfc1_bias
        dfc1_hidden_bn = batch_norm(dfc1_hidden_sum)
        dfc1_hidden = tf.nn.elu(dfc1_hidden_bn)

        dfc2_weight = tf.Variable(tf.truncated_normal([512, 11*11*32], stddev = 0.02))
        dfc2_bias = tf.Variable(tf.constant(0.02, shape = [11*11*32]))
        dfc2_hidden_sum = tf.matmul(dfc1_hidden, dfc2_weight) + dfc2_bias
        dfc2_hidden_bn = batch_norm(dfc2_hidden_sum)
        dfc2_hidden = tf.nn.elu(dfc2_hidden_bn)
        dfc2_hidden_conv = tf.reshape(dfc2_hidden, [-1, 11, 11, 32])

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 11, 11, 32])
        dconv1_hidden_sum = tf.nn.conv2d_transpose(dfc2_hidden_conv, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias
        dconv1_hidden_bn = batch_norm(dconv1_hidden_sum)
        dconv1_hidden = tf.nn.elu(dconv1_hidden_bn)

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [16]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 21, 21, 16])
        dconv2_hidden_sum = tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias
        dconv2_hidden_bn = batch_norm(dconv2_hidden_sum)
        dconv2_hidden = tf.nn.elu(dconv2_hidden_bn)

        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [1]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 84, 84, 1])
        dconv3_hidden_sum = tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias

        return dconv3_hidden_sum, code_layer

    def update_code(self, sess, state):
        previous_state = [x[0] for x in state]
        current_state = [x[1] for x in state]
        sess.run(self.optimize, feed_dict={self.previous_state : previous_state, self.current_state: current_state})

    def get_code(self, state):
        previous_state = [x[0] for x in state]
        current_state = [x[1] for x in state]

        current_codes = []
        change_codes = []
        for i in range(self.code_level):
            current_codes.append(self.current_codes[i].eval(feed_dict={self.current_state: current_state}))
            change_codes.append(self.change_codes[i].eval(feed_dict={self.previous_state : previous_state, self.current_state: current_state}))
        
        result = []
        for i in range(len(state)):
            number = 0
            for c in range(self.code_level):
                for j in range(self.code_size):
                    number *= 2
                    if current_codes[c][i][j] > 0:
                        number += 1
            for c in range(self.code_level):
                for j in range(self.code_size):
                    number *= 2
                    if change_codes[c][i][j] > 0:
                        number += 1
            result.append(number)
        return result

    def print_loss(self, state):
        previous_state = [x[0] for x in state]
        current_state = [x[1] for x in state]
        
        print('total_rec_current_loss', self.total_rec_current_loss.eval(feed_dict={self.current_state: current_state}))
        print('total_rec_change_loss', self.total_rec_change_loss.eval(feed_dict={self.previous_state : previous_state, self.current_state: current_state}))
        print('total_current_code_loss', self.total_current_code_loss.eval(feed_dict={self.current_state: current_state}))
        print('total_change_code_loss', self.total_change_code_loss.eval(feed_dict={self.previous_state : previous_state, self.current_state: current_state}))
