import keras
from predatordata import load_sexual_predator_data
import sys
from ChatLSTM import ChatLSTM, ActivityDirectionRegularizer, default_parameters
from train import fmeasure_objective, f05
import argparse


class Object(object):
    pass


def main(filepath='model.h5'):
    try:
        model = keras.models.load_model(filepath, custom_objects={'ChatLSTM': ChatLSTM,
                                                                  'ActivityDirectionRegularizer': ActivityDirectionRegularizer,
                                                                  'fmeasure_objective': fmeasure_objective, 'f05': f05})
    except:
        print('Model load error!')
        raise
    def_params = default_parameters()
    inputlayer = model.get_layer(name='p1')
    size = inputlayer.input_shape
    tmp = Object()
    tmp.top_word_num = def_params['top_word_num']
    tmp.chat_length = size[1]
    tmp.message_length = size[2]
    _, _, test_data1, test_data2, _, test_label = load_sexual_predator_data(tmp)
    loss, f05_score, precision_score, recall_score = model.evaluate({'p1': test_data1, 'p2': test_data2}, test_label,
                                                                    batch_size=32)

    # print metrics
    print('Results on testdata:')
    print('F0.5: ' + str(f05_score))
    print('Precision: ' + str(precision_score))
    print('Recall: ' + str(recall_score))


if __name__ == '__main__':
    main(sys.argv[1])
