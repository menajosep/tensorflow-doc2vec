from scipy import spatial

import pandas as pd
from nltk.tokenize import word_tokenize

from utils import *

TEXT_WINDOW_SIZE = 8
VOCAB_SIZE = 20000
REMOVE_TOP_K_TERMS = 100

BATCH_SIZE = 10 * TEXT_WINDOW_SIZE
EMBEDDING_SIZE = 128
SHUFFLE_EVERY_X_EPOCH = 5
PV_TEST_SET_PERCENTAGE = 5
NUM_STEPS = 10001
LEARNING_RATE = 0.1
NUM_SAMPLED = 64
REPORT_EVERY_X_STEPS = 2000

END_TO_END_EVERY_X_STEPS = 30000
E2E_TEST_SET_PERCENTAGE = 30
TSNE_NUM_DOCS = 400


def repeater_shuffler(l_):
    l = np.array(l_, dtype=np.int32)
    epoch = 0
    while epoch >= 0:
        if epoch % SHUFFLE_EVERY_X_EPOCH == 0:
            np.random.shuffle(l)
        for i in l:
            yield i
        epoch += 1


def encode_label(row):
    if 0 <= float(row.label_values) < 0.2:
        return int(0)
    elif 0.2 < float(row.label_values) <= 0.4:
        return int(1)
    elif 0.4 < float(row.label_values) <= 0.6:
        return int(2)
    elif 0.6 < float(row.label_values) <= 0.8:
        return int(3)
    elif 0.8 < float(row.label_values) <= 1.0:
        return int(4)
    else:
        print row.label_values


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def get_labels():
    with open('data/stanfordSentimentTreebank/small_sentiment_labels.txt', mode='r') as labels_file:
        orig_labels = pd.read_csv(labels_file, delimiter='|', names=['phrases_ids', 'label_values'],
                                  header=0, encoding='utf8')
        orig_labels['encode_label'] = orig_labels.apply(lambda row: encode_label(row), axis=1)
        orig_labels = dense_to_one_hot(orig_labels.encode_label, 5)
        print len(orig_labels)
        return orig_labels


def build_dictionary():
    with open('data/stanfordSentimentTreebank/small_dictionary.txt', mode='r') as dict_file:
        phrases = pd.read_csv(dict_file, delimiter='|', names=['phrase', 'phrase_index'],
                              encoding='utf8')
        print len(phrases)
        counts = phrases.phrase.apply(lambda x: pd.value_counts(word_tokenize(x))).sum(axis=0)
        max_vocab_size = min(VOCAB_SIZE - 2 + REMOVE_TOP_K_TERMS, counts.shape[0])
        print max_vocab_size
        counts.sort_values(inplace=True, ascending=False)
        counts = counts[REMOVE_TOP_K_TERMS:max_vocab_size]
        counts.set_value('__UNK__', 0)
        counts.set_value('__NULL__', 0)

        dictionary = {}
        for i, word in enumerate(counts.index):
            dictionary[word] = i
        reverse_dictionary = dict(zip(dictionary.values(),
                                      dictionary.keys()))
        vocab_size = len(counts)
        print vocab_size
        del counts

        data = []
        doclens = []
        for docid, (phrase, phrase_index) in enumerate(phrases.values):
            words = word_tokenize(phrase)
            for word in words:
                if word in dictionary:
                    wordid = dictionary[word]
                else:
                    wordid = dictionary['__UNK__']
                data.append((docid, wordid))
            # Pad with NULL values if necessary
            doclen = len(words)
            doclens.append(doclen)
            if doclen < TEXT_WINDOW_SIZE:
                n_nulls = TEXT_WINDOW_SIZE - doclen
                data.extend([(docid, dictionary['__NULL__'])] * n_nulls)

        return dictionary, vocab_size, data, doclens


def get_text_window_center_positions(data):
    # If TEXT_WINDOW_SIZE is even, then define text_window_center
    # as left-of-middle-pair
    doc_start_indexes = [0]
    last_docid = data[0][0]
    for i, (d, _) in enumerate(data):
        if d != last_docid:
            doc_start_indexes.append(i)
            last_docid = d
    twcp = []
    for i in range(len(doc_start_indexes) - 1):
        twcp.extend(list(range(
            doc_start_indexes[i] + (TEXT_WINDOW_SIZE - 1) // 2,
            doc_start_indexes[i + 1] - TEXT_WINDOW_SIZE // 2
        )))
    return twcp

def build_test_twcp(doc, dictionary):
    test_data = []
    words = word_tokenize(str(doc))
    for word in words:
        if word in dictionary:
            wordid = dictionary[word]
        else:
            wordid = dictionary['__UNK__']
        test_data.append((0, wordid))
    # Pad with NULL values if necessary
    doclen = len(words)
    if doclen < TEXT_WINDOW_SIZE:
        n_nulls = TEXT_WINDOW_SIZE - doclen
        test_data.extend([(0, dictionary['__NULL__'])] * n_nulls)
    test_twcp = []
    test_twcp.extend(list(range(
        0 + (TEXT_WINDOW_SIZE - 1) // 2,
        len(test_data) - TEXT_WINDOW_SIZE // 2
    )))
    return test_data, test_twcp

def test_logistic_regression(embeddings, labels):
    global session
    # test with logistic
    W = tf.Variable(tf.zeros([EMBEDDING_SIZE, 5]))
    b = tf.Variable(tf.zeros([5]))
    x = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE])
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 5])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # Split up data set into train/test
    target_train, texts_train, train_indices = get_train_data(labels, embeddings)
    target_test, texts_test = get_test_data(labels, train_indices, embeddings)
    for _ in range(NUM_STEPS):
        rand_index = np.random.choice(target_train.shape[0], size=BATCH_SIZE)
        batch_xs = texts_train[rand_index]
        batch_ys = target_train[rand_index]
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_xs = texts_test
    batch_ys = target_test
    print(session.run(accuracy, feed_dict={x: batch_xs,
                                           y_: batch_ys}))

