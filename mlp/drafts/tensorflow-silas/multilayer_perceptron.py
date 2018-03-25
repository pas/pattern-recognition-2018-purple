import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


batch_size = 100
training_iteration = 30
learning_rate = 0.01


x = tf.placeholder(tf.float32, [None, 784])
y_labels = tf.placeholder(tf.float32, [None, 10])


W_1 = tf.Variable(tf.random_uniform([784, 50]))  # input to hidden
b_1 = tf.Variable(tf.random_uniform([50]))  # hidden

W_2 = tf.Variable(tf.random_uniform([50, 10]))  # hidden to output
b_2 = tf.Variable(tf.random_uniform([10]))  # output


hidden = tf.nn.softmax(tf.matmul(x, W_1) + b_1)
output = tf.nn.softmax(tf.matmul(hidden, W_2) + b_2)


cost_function = -tf.reduce_sum(y_labels * tf.log(output))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for iteration in range(training_iteration):
    avg_loss = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y_labels: batch_ys})

        avg_loss += sess.run(cost_function, feed_dict={x: batch_xs, y_labels: batch_ys}) / total_batch

    print("Epoch {} complete, loss={}".format(iteration + 1, avg_loss))

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_labels: mnist.test.labels}))

