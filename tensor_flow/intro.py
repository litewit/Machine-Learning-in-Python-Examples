import tensorflow as tf


# creates nodes in a graph
# "construction phase"
x1 = tf.constant([[5., 6.],
                  [2., 3.]]
                 )
x2 = tf.constant([[4., 7.],
                  [3., 8.]]
                 )
result = tf.matmul(x1, x2)
print(result)

# defines our session and lanuches graph
with tf.Session() as sess:
    # run result
    output = sess.run(result)
    print(output)
