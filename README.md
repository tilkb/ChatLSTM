# ChatLSTM network for chat representation
End to end deep learning solution for PAN 2012 predator identification Task 1 (http://pan.webis.de/clef12/pan12-web/author-identification.html)
Current solution far from the best solutions, F05 score is about 0.27 on test set.

#Topology
## Chat LSTM-Cell:
2 standard LSTM cell and they are "communicating" with each others
+ Input gate:  i<sub>t</sub>=sigm(W<sup>(i)</sup>x<sub>t</sub>+U<sup>(i)</sup>h<sub>t-1</sub>+b<sup>(i)</sup>)


+ Forget gate: f<sub>t</sub>=sigm(W<sup>(f)</sup>x<sub>t</sub>+U<sup>(f)</sup>h<sub>t-1</sub>+b<sup>(f)</sup>)


+ Output gate: o<sub>t</sub>=sigm(W<sup>(o)</sup>x<sub>t</sub>+U<sup>(o)</sup>h<sub>t-1</sub>+b<sup>(o)</sup>)


+ Response gate: s<sub>t</sub>=sigm(W<sup>(s)</sup>h_<sub>t</sub>+U<sup>(s)</sup>h<sub>t-1</sub>+b<sup>(s)</sup>)


+ u<sub>t</sub>=tanh(W<sup>(u)</sup>x<sub>t</sub>+U<sup>(u)</sup>h<sub>t-1</sub>+b<sup>(u)</sup>)


+ e<sub>t</sub>=tanh(W<sup>(e)</sup>h_<sub>t</sub>+U<sup>(e)</sup>h<sub>t-1</sub>+b<sup>(e)</sup>)


+ c<sub>t</sub>=u<sub>t</sub> \* i<sub>t</sub>+c<sub>t-1</sub> \* f <sub>t</sub>+e<sub>t</sub> \* s<sub>t</sub>


+ h<sub>t</sub>=o<sub>t</sub> \* tanh(c<sub>t</sub>)

h_ is the other LSTM's hidden state

## Hierarchical stucture:
Message is vectorized with LSTM.
Each message pairs' vectors are merged, these vectors are the input of the chatLSTM.

![Network topology](network.png?raw=true "Network topology")

## Regularization
Direction Regularization: At vector representation of the chat 2 part of the vector represents each speaker's intention, these vectors should have the same direction, if they think the same about the topic. That is why two vector's normalized cos distance is used as regularization in the loss function.

## F-0.5 measure optimization
The dataset is very unbalanced and F0.5 metric used for evaluate the model, so basic cross-entropy isn't good enough, because it optimize for balanced classes' accuracy.
F measures are harmonic average of recall and precision:
Modified objective:
y=predicted
z=label
'(1+alfa\*z)\*cross-entropy(y,z)+(1+alfa\*z)\*beta\*log(y)'
+ First part optimize for recall: higher alfa cause higher recall
+ Second part cause higher precision, if beta is higher


# Command line usage
## Train
'python3 train.py'

Parameters
+ Number of words are in the dictionary: '--top_word_num=5000'
+ Dimension of the word representation: '--word_dim='
+ Dimension of the message representation: '--hidden_dim1'
+ Dimension of the chat representation: '--hidden_dim2='
+ Number of neurons in the first fully connected layer: '--dense_hidden='
+ Dropout probability in fully connected layer: '--dropout=0.5'
+ Batch size: '--batch_size=128'
+ Number of epochs: '--max_epoch_number=150'
+ Size of the validation data  '--validation_size=0.1'
+ Use ChatLSTM or basic LSTM for chat representation: '--use_chat_LSTM=True'
+ Use direction regularization: '--direction_reg_on=True'
+ Weight of the direction regulation: 'direction_reg=0.1'
+ Message length in words used for bucket: 'message_length=40'
+ Chat length used for bucket: 'chat_length=40'
+ Weight of the recall: 'recall_parameter=45'
+ Weight of the precision, 0 means no change: 'precision_parameter=0.05'

Best models are saved to tmp folder.

##Test the saved models
'python3 test.py filename'
