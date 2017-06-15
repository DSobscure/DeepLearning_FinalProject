import tensorflow as tf

class SCG():
    def __init__(self, code_size):
        self.code_size = code_size
        self.state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='state')
        self.next_state = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.float32, name='next_state')
        self.state_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='state_code')
        self.difference_code = tf.placeholder(shape=[None, code_size], dtype=tf.float32, name='difference_code')
        self.code_output, self.code_output2 = self.build_network(self.state, self.next_state, trainable=True)    
        self.difference_code_output, self.difference_code_output2 = self.build_network(self.next_state - self.state, self.next_state - self.state, trainable=True)            

        self.code_loss = tf.reduce_mean(tf.pow(self.code_output - self.state_code, 2))
        self.code_loss2 = tf.reduce_mean(tf.pow(self.code_output - self.code_output2, 2))
        self.optimize_code = tf.train.RMSPropOptimizer(0.00025).minimize(self.code_loss + self.code_loss2)   

        self.difference_code_loss = tf.reduce_mean(tf.pow(self.difference_code_output - self.difference_code, 2))
        self.optimize_difference_code = tf.train.RMSPropOptimizer(0.00025).minimize(self.difference_code_loss)   

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

        fc1_weight = tf.Variable(tf.truncated_normal([14*14*32, self.code_size], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 14*14*32])
        conv3_hidden_flat2 = tf.reshape(conv3_hidden2, [-1, 14*14*32])
        fc1_hidden_sum = tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias
        fc1_hidden = tf.nn.sigmoid(fc1_hidden_sum)
        fc1_hidden_sum2 = tf.matmul(conv3_hidden_flat2, fc1_weight) + fc1_bias
        fc1_hidden2 = tf.nn.sigmoid(fc1_hidden_sum2)

        print("code layer shape : %s" % fc1_hidden.get_shape())

        return fc1_hidden, fc1_hidden2

    def update_code(self, sess, state, code, next_state, difference_code):
        sess.run([self.optimize_code, self.optimize_difference_code], feed_dict={self.state : state, self.state_code: code, self.next_state: next_state, self.difference_code: difference_code})

    def get_code(self, state, next_state):
        outputs = self.code_output.eval(feed_dict={self.state: state})
        difference_outputs = self.difference_code_output.eval(feed_dict={self.state: state, self.next_state: next_state})
        result = 0
        for i in range(len(outputs)):
            for j in range(self.code_size):
                result *= 2
                if outputs[i][j] > 0.5:
                    result += 1
            for j in range(self.code_size):
                result *= 2
                if difference_outputs[i][j] > 0.5:
                    result += 1
        return result

    def get_code_batch(self, state, next_state):
        outputs = self.code_output.eval(feed_dict={self.state: next_state})
        difference_outputs = self.difference_code_output.eval(feed_dict={self.state: state, self.next_state: next_state})
        result = []
        for i in range(len(outputs)):
            number = 0
            for j in range(self.code_size):
                number *= 2
                if outputs[i][j] > 0.5:
                    number += 1
            for j in range(self.code_size):
                number *= 2
                if difference_outputs[i][j] > 0.5:
                    number += 1
            result.append(number)
        return result


