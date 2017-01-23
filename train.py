import keras
from keras.models import Model
from ChatLSTM import ChatLSTM, default_parameters, ActivityDirectionRegularizer
from predatordata import load_sexual_predator_data
from keras.metrics import fbeta_score
from keras.engine import InputSpec
from keras.layers import Input, Dense, Embedding, TimeDistributed, LSTM, Activation, Dropout
from keras import backend as K
import argparse
from keras.callbacks import ModelCheckpoint, CSVLogger
# custom metrics
def f05(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5)

#custom objective to optimize F0.5 measure
def fmeasure_objective(y_true, y_pred):
    return K.binary_crossentropy(y_pred,y_true)+(alfa * y_true) * K.binary_crossentropy(y_pred,y_true) + beta * K.log(y_pred) + (alfa * y_true) * beta * K.log (y_pred)

def buildmodel(parameters):
    # build the model from components, with the given parameters,
    # first person's messages
    inputs1 = Input(shape=(parameters.chat_length, parameters.message_length), name='p1')
    # second person's messages
    inputs2 = Input(shape=(parameters.chat_length, parameters.message_length), name='p2')
    embeding = Embedding(parameters.top_word_num, parameters.word_dim)
    emb1 = TimeDistributed(embeding)(inputs1)
    emb2 = TimeDistributed(embeding)(inputs2)
    # message encoder for both person
    textencoderlstm = LSTM(parameters.hidden_dim1)
    encodermessage1 = TimeDistributed(textencoderlstm)(emb1)
    encodermessage2 = TimeDistributed(textencoderlstm)(emb2)
    # connect 2 message to message pair
    messagepair_encoder = keras.layers.merge([encodermessage1, encodermessage2], mode='concat')
    if ( parameters.use_chat_LSTM == False):
        chatencoder = LSTM(parameters.hidden_dim2)(messagepair_encoder)  # basiclstm
    else:
        chatencoder = ChatLSTM(parameters.hidden_dim2)(messagepair_encoder)  # chatlstm

    #vector direction regularization
    if (parameters.direction_reg_on == True):
        regularizer = (ActivityDirectionRegularizer(parameters.batch_size, alfa=parameters.direction_reg))(
            chatencoder)
        # classification
        s1 = Dense(parameters.dense_hidden)(regularizer)
    else:
        s1 = Dense(parameters.dense_hidden)(chatencoder)
    h1 = Activation('relu')(s1)
    dropout = Dropout(parameters.dropout)(h1)
    s2 = Dense(1)(dropout)
    out = Activation('sigmoid')(s2)
    model = Model(input=[inputs1, inputs2], output=[out])
    model.compile(optimizer='rmsprop', loss=fmeasure_objective, metrics=[f05,'precision', 'recall'])
    return model
def parse():
    parser = argparse.ArgumentParser()
    def_params=default_parameters()
    parser.add_argument('--top_word_num', type=int,
                        default=def_params['top_word_num'],
                        help="Number of different word in the chats")
    parser.add_argument('--word_dim', type=int,
                        default=def_params['word_dim'],
                        help="Dimension of embedding words")
    parser.add_argument('--hidden_dim1', type=int,
                        default=def_params['hidden_dim1'],
                        help="Dimension of message representation")
    parser.add_argument('--hidden_dim2', type=int,
                        default=def_params['hidden_dim2'],
                        help="Dimension of chat representation")
    parser.add_argument('--dense_hidden', type=int,
                        default=def_params['dense_hidden'],
                        help="Dimension of the first fully connected layer in the classificator part of the net")
    parser.add_argument('--dropout', type=float,
                        default=def_params['dropout'],
                        help="Dropout in the fully connected layer")
    parser.add_argument('--batch_size', type=int,
                        default=def_params['batch_size'],
                        help="Size of the minibatch")
    parser.add_argument('--max_epoch_number', type=int,
                        default=def_params['max_epoch_number'],
                        help="Number of max epoch number")
    parser.add_argument('--validation_size', type=float,
                        default=def_params['validation_size'],
                        help="Size of the validation data [0..1]")
    parser.add_argument('--use_chat_LSTM', type=bool,
                        default=def_params['use_chat_LSTM'],
                        help="Type of LSTM used for chat representation: True=ChatLSTM/False=LSTM")
    parser.add_argument('--direction_reg_on', type=bool,
                        default=def_params['direction_reg_on'],
                        help="Use of direction regularizer as additional cost of the cosinus distance")
    parser.add_argument('--direction_reg', type=float,
                        default=def_params['dense_hidden'],
                        help="Weight of the direction regularizer")
    #Data preprocessing options:
    parser.add_argument('--message_length', type=int,
                        default=def_params['message_length'],
                        help="Bucket size(message length) for messages")
    parser.add_argument('--chat_length', type=int,
    default = def_params['chat_length'],
              help = "Bucket size(chat length) for messages")
    #objective parameters
    parser.add_argument('--recall_parameter', type=int,
                        default=def_params['recall_parameter'],
                        help="")
    # objective parameters
    parser.add_argument('--precision_parameter', type=int,
                        default=def_params['precision_parameter'],
                        help="")


    args = parser.parse_args()
    return args


def main():
    parameters = parse()
    global alfa
    alfa = parameters.recall_parameter
    global beta
    beta = parameters.precision_parameter
    #load predator data
    print("Loading data...")
    train_data1, train_data2, test_data1, test_data2, train_label, test_label = load_sexual_predator_data(parameters)
    print("Build model...")
    model=buildmodel(parameters)
    checkpoint = ModelCheckpoint("tmp/model-{val_f05:.4f}-{epoch}.h5", monitor='val_f05', verbose=1,
                                 save_best_only=True, mode='max')
    logging = CSVLogger('tmp/train.csv', separator=',', append=True)

    model.fit({'p1': train_data1, 'p2': train_data2}, train_label, batch_size=parameters.batch_size, nb_epoch=parameters.max_epoch_number, verbose=1, validation_split=parameters.validation_size, callbacks=[checkpoint, logging])
    loss, f05_score,precision_score, recall_score = model.evaluate( {'p1': test_data1, 'p2': test_data2}, test_label, batch_size=parameters.batch_size)

    #print metrics
    print('Results on testdata:')
    print('F0.5: ' + str(f05_score))
    print('Precision: ' + str(precision_score))
    print('Recall: ' + str(recall_score))
    try:
        model.save('predator_model-' + str(f05_score) + '.h5')
    except:
        print('Save error!')

if __name__ == '__main__':
    main()