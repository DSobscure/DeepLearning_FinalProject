import tensorflow as tf

CODE_SIZE = 64

class CodeGenerator(object):
    def __init__(self):
        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
        self.next_state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
        self.state_code, self.next_state_code = self.build_network(self.state, self.next_state, trainable=True)
        self.loss = -tf.reduce_mean(tf.pow(self.state_code - self.next_state_code, 2))
        self.optimize_op = tf.train.RMSPropOptimizer(0.000025, momentum=0.95, epsilon=0.01).minimize(self.loss)    

    def build_network(self, state, next_state, trainable=True):
        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        
        conv1_hidden = tf.nn.relu(tf.nn.conv2d(state, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)
        conv1_hidden2 = tf.nn.relu(tf.nn.conv2d(next_state, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias)

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        
        conv2_hidden = tf.nn.relu(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)
        conv2_hidden2 = tf.nn.relu(tf.nn.conv2d(conv1_hidden2, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias)

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        
        conv3_hidden = tf.nn.relu(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)
        conv3_hidden2 = tf.nn.relu(tf.nn.conv2d(conv2_hidden2, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias)

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        conv3_hidden_flat2 = tf.reshape(conv3_hidden2, [-1, 11*11*64])
        fc1_hidden = tf.nn.relu(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias)
        fc1_hidden2 = tf.nn.relu(tf.matmul(conv3_hidden_flat2, fc1_weight) + fc1_bias)

        fc2_weight = tf.Variable(tf.truncated_normal([512, CODE_SIZE], stddev = 0.02), trainable = trainable)
        fc2_bias = tf.Variable(tf.constant(0.02, shape = [CODE_SIZE]), trainable = trainable)
        fc2_hidden = tf.nn.tanh(tf.matmul(fc1_hidden, fc2_weight) + fc2_bias)
        fc2_hidden2 = tf.nn.tanh(tf.matmul(fc1_hidden2, fc2_weight) + fc2_bias)

        print("code layer shape : %s" % fc2_hidden.get_shape())

        return fc2_hidden, fc2_hidden2

    def update(self, sess, state, next_state):
        loss, _ = sess.run([self.loss, self.optimize_op], feed_dict={self.state : state, self.next_state: next_state})

    def get_codes(self, sess, states):
        codeBatch = sess.run(self.state, feed_dict={self.state : states})
        result = []
        for i in range(BATCH_SIZE):
            number = 0
            for j in range(CODE_SIZE):
                number *= 2
                if codeBatch[i][j] > 0:
                    number += 1
            result.append(number)
            codeSet.add(number)
        return result


