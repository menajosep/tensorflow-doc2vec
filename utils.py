import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# Network Parameters
n_hidden_1 = 128  # 1st layer number of features
n_hidden_2 = 128  # 2nd layer number of features
n_classes = 128


# Create model for each network
def build_nn(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def set_weights_and_biases(max_features):
    # Initialize placeholders
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([max_features / 2, n_hidden_1], dtype=tf.float64), dtype=tf.float64),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=tf.float64), dtype=tf.float64),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], dtype=tf.float64), dtype=tf.float64)
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64), dtype=tf.float64),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=tf.float64), dtype=tf.float64),
        'out': tf.Variable(tf.random_normal([n_classes], dtype=tf.float64), dtype=tf.float64)
    }
    return biases, weights


def get_train_data(target, training_set):
    train_indices = np.random.choice(training_set.shape[0], int(round(0.8 * training_set.shape[0])),
                                     replace=False)
    texts_train = training_set[train_indices]
    target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
    return target_train, texts_train, train_indices


def get_test_data(target, train_indices, training_set):
    test_indices = np.array(list(set(range(training_set.shape[0])) - set(train_indices)))
    texts_test = training_set[test_indices]
    target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])
    return target_test, texts_test


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries_' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)


def record_loss_and_accuracy(accuracy, i, i_data, loss, max_features, rand_x, rand_y, sess, target_test, test_acc,
                             test_loss, texts_test, train_acc, train_loss, x1_data, x2_data, y_target, model_output,
                             distances, merged, train_writer):
    # Only record loss and accuracy every 100 generations
    if (i + 1) % 50 == 0:
        i_data.append(i + 1)
        [summary, train_loss_temp, train_acc_temp, distance_temp] = sess.run([merged, loss, accuracy, model_output],
                                                                             feed_dict={x1_data: rand_x[:,
                                                                                                 :max_features / 2],
                                                                                        x2_data: rand_x[:,
                                                                                                 max_features / 2:max_features],
                                                                                        y_target: rand_y})
        train_loss.append(train_loss_temp)
        train_acc.append(train_acc_temp)
        distances.append(distance_temp)
        train_writer.add_summary(summary, i)

        dense_texts_test = texts_test.todense()

        [test_loss_temp, test_acc_temp] = sess.run([loss, accuracy],
                                                   feed_dict={x1_data: dense_texts_test[:, :max_features / 2],
                                                              x2_data: dense_texts_test[:,
                                                                       max_features / 2:max_features],
                                                              y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)
        test_acc.append(test_acc_temp)
    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print(
            'Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
                *acc_and_loss))


def print_loss_and_accuracy(i_data, test_acc, test_loss, train_acc, train_loss, model_type, output, iterations,
                            distances):
    # Plot loss over time
    plt.plot(i_data, train_loss, 'k-', label='Train Loss')
    plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig(
        output + '/' + model_type + '_' + str(iterations) + '_deep_class_loss_' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%SUTC') + ".png")
    plt.clf()

    # Plot train and test accuracy
    plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.draw()
    plt.savefig(
        output + '/' + model_type + '_' + str(iterations) + '_deep_class_acc_' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%SUTC') + ".png")
    plt.clf()

    # Plot train and test accuracy
    distances_array = np.array(distances).flatten()
    plt.hist(distances_array[np.logical_not(np.isnan(distances_array))], bins=100)
    plt.title('Distance')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.draw()
    plt.savefig(
        output + '/' + model_type + '_' + str(
            iterations) + '_deep_class_distance_' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%SUTC') + ".png")
