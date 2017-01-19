try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import sys
import os
import shutil
import zipfile
import gzip
import numpy as np
import xml.etree.ElementTree as ET
import keras

#  load sexual predator data-PAN2012
def load_sexual_predator_data(parameters):
    if not os.path.exists('data'):
        download()
    # filenames
    trainingdata = "data/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
    traininglabel = "data/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
    testdata = "data/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
    testlabel = "data/pan12-sexual-predator-identification-groundtruth-problem1.txt"
    # load labels
    train_positive = []
    test_positive = []
    f = open(traininglabel, 'r')
    for tmp in f:
        train_positive.append(str(tmp).rstrip())
    f.close()
    f = open(testlabel, 'r')
    for tmp in f:
        test_positive.append(str(tmp).rstrip())
    f.close()

    chatdata1_train = []
    chatdata2_train = []
    labels_train = []
    chatdata1_test = []
    chatdata2_test = []
    labels_test = []
    # train
    tree = ET.parse(trainingdata)
    root = tree.getroot()
    jj = 0
    for conversation in root:
        # speaker ids
        p1 = None
        p2 = None
        # last speaker
        last = 1
        # lenght of the chat
        chat_len = 1
        msg = ''
        chat1 = []
        chat2 = []
        for message in conversation:
            author = message[0].text
            text = str(message[2].text)
            if (p1 == None):
                p1 = author
            else:
                if (p2 == None):
                    p2 = author
            # current speaker bit
            current = 1
            if (chat_len == parameters.chat_length * 2 + 1):
                break
            if (author == p1):
                current = 0

            # when change the speaker-»save it
            if (current != last and chat_len != 1):
                vectors = keras.preprocessing.text.one_hot(msg, parameters.top_word_num, lower=True, split=" ")
                if (chat_len % 2 == 1):
                    chat1.append(vectors)
                else:
                    chat2.append(vectors)
                msg = str(text)
                chat_len = chat_len + 1
            else:
                msg = msg + ' ' + str(text)
            last = current
        # last message:
        if (chat_len < parameters.chat_length * 2):
            vectors = keras.preprocessing.text.one_hot(msg, parameters.top_word_num, lower=True, split=" ")
            if (chat_len % 2 == 1):
                chat1.append(vectors)
            else:
                chat2.append(vectors)
            chat_len += 1

            # insert empty messages, if it needs place
        for j in range(chat_len, parameters.chat_length * 2 + 1):
            if (j % 2 == 1):
                chat1.insert(0, [0])
            else:
                chat2.insert(0, [0])

        # put pad symbol at the begining of the sequence
        v1 = keras.preprocessing.sequence.pad_sequences(chat1, maxlen=parameters.message_length, padding='pre',
                                                        truncating='pre', value=0, dtype='int32')
        v2 = keras.preprocessing.sequence.pad_sequences(chat2, maxlen=parameters.message_length, padding='pre',
                                                        truncating='pre', value=0, dtype='int32')
        chatdata1_train.append(np.array(v1))
        chatdata2_train.append(np.array(v2))
        jj += 1

        # is any of the predator
        if (p1 in train_positive or p2 in train_positive):
            labels_train.append(1)
        else:
            labels_train.append(0)
    # put data to numpy array, this way every dimension of the array is visible
    ch1_train = np.zeros((jj, parameters.chat_length, parameters.message_length))
    ch2_train = np.zeros((jj, parameters.chat_length, parameters.message_length))
    final_labels_train = np.zeros((jj))
    for item in range(jj):
        ch1_train[item] = chatdata1_train[item]
        ch2_train[item] = chatdata2_train[item]
        final_labels_train[item] = labels_train[item]
    # --------------TEST DATA---------------------------
    tree = ET.parse(testdata)
    root = tree.getroot()
    jj = 0
    for conversation in root:
        # speaker ids
        p1 = None
        p2 = None
        # last speaker
        last = 1
        # lenght of the chat
        chat_len = 1
        msg = ''
        chat1 = []
        chat2 = []
        for message in conversation:
            author = message[0].text
            text = message[2].text
            if (p1 == None):
                p1 = author
            else:
                if (p2 == None):
                    p2 = author
            # current speaker bit
            current = 1
            if (chat_len == parameters.chat_length * 2 + 1):
                break
            if (author == p1):
                current = 0

            # when change the speaker-»save it
            if (current != last and chat_len != 1):
                chat_len = chat_len + 1
                vectors = keras.preprocessing.text.one_hot(msg, parameters.top_word_num, lower=True, split=" ")
                if (chat_len % 2 == 1):
                    chat1.append(vectors)
                else:
                    chat2.append(vectors)
                msg = str(text)
            else:
                msg = msg + ' ' + str(text)
            last = current
        # last message:

        if (chat_len < parameters.chat_length * 2):
            vectors = keras.preprocessing.text.one_hot(msg, parameters.top_word_num, lower=True, split=" ")
            if (chat_len % 2 == 1):
                chat1.append(vectors)
            else:
                chat2.append(vectors)
            chat_len += 1
        # insert empty messages, if it needs place
        for j in range(chat_len, parameters.chat_length * 2 + 1):
            if (j % 2 == 1):
                chat1.insert(0, [0])
            else:
                chat2.insert(0, [0])
        # put pad symbol at the begining of the sequence
        v1 = keras.preprocessing.sequence.pad_sequences(chat1, maxlen=parameters.message_length, padding='pre',
                                                        truncating='pre', value=0, dtype='int32')
        v2 = keras.preprocessing.sequence.pad_sequences(chat2, maxlen=parameters.message_length, padding='pre',
                                                        truncating='pre', value=0, dtype='int32')
        chatdata1_test.append(np.array(v1))
        chatdata2_test.append(np.array(v2))
        jj += 1

        # is any of the predator
        if (p1 in test_positive or p2 in test_positive):
            labels_test.append(1)
        else:
            labels_test.append(0)
    # put data to numpy array, this way every dimension of the array is visible
    ch1_test = np.zeros((jj, parameters.chat_length, parameters.message_length))
    ch2_test = np.zeros((jj, parameters.chat_length, parameters.message_length))
    final_labels_test = np.zeros((jj))
    for item in range(jj):
        ch1_test[item] = chatdata1_test[item]
        ch2_test[item] = chatdata2_test[item]
        final_labels_test[item] = labels_test[item]

    return ch1_train, ch2_train, ch1_test, ch2_test, final_labels_train, final_labels_test

def download():
    trainurl='http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-12/pan12-data/pan12-sexual-predator-identification-training-corpus-2012-05-01.zip'
    testurl='http://www.uni-weimar.de/medien/webis/corpora/corpus-pan-labs-09-today/pan-12/pan12-data/pan12-sexual-predator-identification-test-corpus-2012-05-21.zip'
    #create directory
    directory = 'data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    traindata = os.path.join(directory, 'train.zip')
    testdata = os.path.join(directory, 'test.zip')
    u1 = urllib2.urlopen(trainurl)
    u2 = urllib2.urlopen(testurl)

    f1 = open(traindata, 'wb')
    f2 = open(testdata, 'wb')
    #download train data
    print("Downloading: %s" % (traindata))
    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u1.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f1.write(buf)
        sys.stdout.flush()
    f1.close()
    #download test data
    print("Downloading: %s " % (traindata))
    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u2.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f2.write(buf)
        sys.stdout.flush()
    f2.close()
    #unzip files
    print("Extracting...")
    with zipfile.ZipFile(traindata) as zf:
        zf.extractall(directory)
    with zipfile.ZipFile(testdata) as zf:
        zf.extractall(directory)
    print("Remove files")
    os.remove(traindata)
    os.remove(testdata)
    shutil.move(os.path.join(directory+'/pan12-sexual-predator-identification-test-corpus-2012-05-21', 'pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'),
                             os.path.join(directory,'pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'))
    shutil.move(os.path.join(directory + '/pan12-sexual-predator-identification-test-corpus-2012-05-21','pan12-sexual-predator-identification-groundtruth-problem1.txt'),
                             os.path.join(directory, 'pan12-sexual-predator-identification-groundtruth-problem1.txt'))

    shutil.move(os.path.join(directory + '/pan12-sexual-predator-identification-training-corpus-2012-05-01',
                             'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'),
                             os.path.join(directory,
                                          'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'))
    shutil.move(os.path.join(directory + '/pan12-sexual-predator-identification-training-corpus-2012-05-01', 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'),
                            os.path.join(directory, 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'))

    shutil.rmtree(os.path.join(directory, 'pan12-sexual-predator-identification-training-corpus-2012-05-01'), ignore_errors=True)
    shutil.rmtree(os.path.join(directory, 'pan12-sexual-predator-identification-test-corpus-2012-05-21'), ignore_errors=True)


