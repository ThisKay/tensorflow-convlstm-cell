import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    batch_size = 10
    timesteps = 5
    shape = [20, 20]
    kernel = [3, 3]
    channels = 1
    filters = 12

    # Create a placeholder for videos.
    # [32, 100, 640, 480, 3]
    inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
    y = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])

    # Add the ConvLSTM step.
    from cell import ConvLSTMCell

    cell = ConvLSTMCell(shape, filters, kernel)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

    outputs = tf.reshape(outputs, [-1] + shape + [filters])
    cvtW = tf.get_variable('cvtkel', [3, 3, filters, 1])
    predData = tf.nn.convolution(outputs, cvtW, 'SAME', data_format='NHWC')
    predData = tf.reshape(predData, [batch_size, timesteps] + shape + [1])

    loss = tf.reduce_mean(tf.square(predData - inputs))
    lr_rate = tf.Variable(0.8, trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    lr_update = tf.assign(lr_rate, new_lr)

    # optimzer = tf.train.AdadeltaOptimizer(lr_rate).minimize(loss)
    optimzer = tf.train.RMSPropOptimizer(lr_rate).minimize(loss)

    # # There's also a ConvGRUCell that is more memory efficient.
    # from cell import ConvGRUCell
    # cell = ConvGRUCell(shape, filters, kernel)
    # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
    #
    #
    # # It's also possible to enter 2D input or 4D input instead of 3D.
    # shape = [100]
    # kernel = [3]
    # inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
    # cell = ConvLSTMCell(shape, filters, kernel)
    # outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
    #
    # shape = [50, 50, 50]
    # kernel = [1, 3, 5]
    # inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
    # cell = ConvGRUCell(shape, filters, kernel)
    # outputs, state= tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

    # more code
    # 输出维度为[b32, t100, w640, h480, c=m=filter]
    np_one = np.ones([10, 1, 20, 20, 1])
    simulate_input = np.concatenate((np_one, np_one * 2), axis=1)
    simulate_input = np.concatenate((simulate_input, np_one * 3), axis=1)
    simulate_input = np.concatenate((simulate_input, np_one * 4), axis=1)
    simulate_input = np.concatenate((simulate_input, np_one * 5), axis=1)
    simulate_y = simulate_input * 2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(lr_update, feed_dict={new_lr: 0.8 ** (i / 10)})
            # inp = np.random.normal(size=(10, 5, 20, 20, 1))

            _, l, o, s = sess.run([optimzer, loss, predData, state], feed_dict={inputs: simulate_input, y: simulate_y})
            # print("out.shape", o.shape)  # (5,2,3,3,6)
            print("loss:", l)

        pass
