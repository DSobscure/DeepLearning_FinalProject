import tensorflow as tf

class SCG():
    def __init__(self, code_size):
        self.code_size = code_size

        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.random_code = tf.placeholder(shape=[None, self.code_size], dtype=tf.float32, name='random_code')

        self.state_code = self.build_network(self.state, trainable=True)            
        self.state_code_loss = tf.reduce_mean(tf.pow(self.state_code - self.random_code, 2))

        self.optimize = tf.train.RMSPropOptimizer(0.00025).minimize(self.state_code_loss)      

    def build_network(self, x, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv1_hidden = tf.nn.relu(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden = tf.nn.relu(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv3_hidden = tf.nn.relu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)

        output_weight = tf.Variable(tf.truncated_normal([512, self.code_size], stddev = 0.02), trainable = trainable)
        output_bias = tf.Variable(tf.constant(0.02, shape = [self.code_size]), trainable = trainable)
        output_sum = tf.matmul(fc1_hidden, output_weight) + output_bias
        output = tf.nn.sigmoid(output_sum)

        print("code layer shape : %s" % output.get_shape())

        return output

    def update_code(self, sess, state, random_code):
        state_code_loss, _ = sess.run([self.state_code_loss, self.optimize], feed_dict={self.state: state, self.random_code: random_code})
        return state_code_loss

    def get_code(self, state):
        state_code = self.state_code.eval(feed_dict={self.state: state})
        
        result = []
        for i in range(len(state)):
            number = 0
            for j in range(self.code_size):
                number *= 2
                if state_code[i][j] > 0.5:
                    number += 1
            result.append(number)
        return result

