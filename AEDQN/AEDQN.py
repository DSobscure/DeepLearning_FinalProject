import tensorflow as tf
import numpy as np

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=True)

class AEDQN():
    def __init__(self, action_number, GAMMA):
        self.action_number = action_number

        self.state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='state')
        self.next_state = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32, name='next_state')
        self.action = tf.placeholder(tf.int32, shape=[None, 1], name='action')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')            
        self.terminal_mask = tf.placeholder(tf.float32, shape=[None, 1], name='terminal_mask')

        self.target_output, self.target_rec_state, self.target_model = self.build_network(self.state, trainable=True)            
        self.beheavior_output, self.beheavior_rec_state, self.beheavior_model = self.build_network(self.next_state, trainable=False)            

        action_mask = tf.one_hot(self.action, self.action_number, name='action_mask')
        action_mask = tf.reshape(action_mask, (-1, self.action_number))    
        masked_target_q = tf.reduce_sum(action_mask * self.target_output, reduction_indices=1)    
        masked_target_q = tf.reshape(masked_target_q, (-1, 1))    
        max_next_q = tf.reduce_max(self.beheavior_output, reduction_indices=1)    
        max_next_q = tf.reshape(max_next_q, (-1, 1))    
        self.delta = self.reward + self.terminal_mask * GAMMA * max_next_q - masked_target_q    
            
        self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')           
        self.train_step = tf.train.RMSPropOptimizer(0.000025).minimize(self.loss)       

    def build_network(self, x, trainable=True):
        variables = dict()
        weight_counter = 1

        conv1_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        conv1_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        conv1_hidden = tf.nn.relu(batch_norm(tf.nn.conv2d(x, conv1_weight, strides = [1,4,4,1], padding='SAME') + conv1_bias))

        variables[weight_counter] = conv1_weight
        weight_counter += 1
        variables[weight_counter] = conv1_bias
        weight_counter += 1

        conv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        conv2_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv2_hidden = tf.nn.relu(batch_norm(tf.nn.conv2d(conv1_hidden, conv2_weight, strides = [1,2,2,1], padding='SAME') + conv2_bias))

        variables[weight_counter] = conv2_weight
        weight_counter += 1
        variables[weight_counter] = conv2_bias
        weight_counter += 1

        conv3_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        conv3_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        conv3_hidden = tf.nn.relu(batch_norm(tf.nn.conv2d(conv2_hidden, conv3_weight, strides = [1,1,1,1], padding='SAME') + conv3_bias))

        variables[weight_counter] = conv3_weight
        weight_counter += 1
        variables[weight_counter] = conv3_bias
        weight_counter += 1

        dconv1_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.02), trainable = trainable)
        dconv1_bias = tf.Variable(tf.constant(0.02, shape = [64]), trainable = trainable)
        dconv1_output_shape = tf.stack([tf.shape(x)[0], 11, 11, 64])
        dconv1_hidden_sum = tf.nn.conv2d_transpose(conv3_hidden, dconv1_weight, dconv1_output_shape, strides = [1,1,1,1], padding='SAME') + dconv1_bias
        dconv1_hidden_bn = batch_norm(dconv1_hidden_sum)
        dconv1_hidden = tf.nn.relu(dconv1_hidden_bn)

        variables[weight_counter] = dconv1_weight
        weight_counter += 1
        variables[weight_counter] = dconv1_bias
        weight_counter += 1

        dconv2_weight = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.02), trainable = trainable)
        dconv2_bias = tf.Variable(tf.constant(0.02, shape = [32]), trainable = trainable)
        dconv2_output_shape = tf.stack([tf.shape(x)[0], 21, 21, 32])
        dconv2_hidden_sum = tf.nn.conv2d_transpose(dconv1_hidden, dconv2_weight, dconv2_output_shape, strides = [1,2,2,1], padding='SAME') + dconv2_bias
        dconv2_hidden_bn = batch_norm(dconv2_hidden_sum)
        dconv2_hidden = tf.nn.relu(dconv2_hidden_bn)

        variables[weight_counter] = dconv2_weight
        weight_counter += 1
        variables[weight_counter] = dconv2_bias
        weight_counter += 1

        dconv3_weight = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.02), trainable = trainable)
        dconv3_bias = tf.Variable(tf.constant(0.02, shape = [4]), trainable = trainable)
        dconv3_output_shape = tf.stack([tf.shape(x)[0], 84, 84, 4])
        dconv3_hidden_sum = tf.nn.conv2d_transpose(dconv2_hidden, dconv3_weight, dconv3_output_shape, strides = [1,4,4,1], padding='SAME') + dconv3_bias

        variables[weight_counter] = dconv3_weight
        weight_counter += 1
        variables[weight_counter] = dconv3_bias
        weight_counter += 1

        fc1_weight = tf.Variable(tf.truncated_normal([11*11*64, 512], stddev = 0.02), trainable = trainable)
        fc1_bias = tf.Variable(tf.constant(0.02, shape = [512]), trainable = trainable)
        conv3_hidden_flat = tf.reshape(conv3_hidden, [-1, 11*11*64])
        fc1_hidden = tf.nn.relu(batch_norm(tf.matmul(conv3_hidden_flat, fc1_weight) + fc1_bias))

        variables[weight_counter] = fc1_weight
        weight_counter += 1
        variables[weight_counter] = fc1_bias
        weight_counter += 1

        linear_weight = tf.Variable(tf.truncated_normal([512, self.action_number], stddev = 0.02), trainable = trainable)
        linear_bias = tf.Variable(tf.constant(0.02, shape = [self.action_number]), trainable = trainable)
        output = tf.matmul(fc1_hidden, linear_weight) + linear_bias

        variables[weight_counter] = linear_weight
        weight_counter += 1
        variables[weight_counter] = linear_bias
        weight_counter += 1

        return output, dconv3_hidden_sum, variables

    def get_values(self, sess, state_batch):
        q = sess.run(self.beheavior_output, feed_dict={self.next_state : state_batch})
        return q

    def select_action(self, sess, state):
        values = self.get_values(sess, [state])[0]
        return np.argmax(values);

    def update(self, sess, state, action, reward, next_state, terminal):
        action = np.reshape(action, (-1, 1))
        reward = np.reshape(reward, (-1, 1))
        terminal = np.reshape(terminal, (-1, 1))

        terminal_mask = np.invert(terminal) * 1

        loss, _ = sess.run([self.loss, self.train_step], feed_dict={
            self.state : state,
            self.next_state : next_state,
            self.action : action,
            self.reward : reward,
            self.terminal_mask : terminal_mask})
        return loss

    def update_target_network(self, sess):
        updates = []
        for key, value in self.target_model.items():
            updates.append(self.beheavior_model[key].assign(value))
        sess.run(updates);