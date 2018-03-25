# An Improved Deep-Learning Model of Turn-taking in Spoken Dialogue

Many turn-taking models have been built to solve specific tasks, like
predicting if a user will hold his/her turn after a pause. However,
Skantze showed that a continuous, general
turn-taking model can handle multiple tasks successfully. Building on
his model, a Long Short-Term Memory (LSTM) Recurrent Neural Network,
we made four improvements: we 1) replaced the Sigmoid Units with
Parametric Rectified Linear Units, 2) increased the number of layers
and neurons, 3) trained without truncation, and 4) added
dropout regularization.  Our improved model gives good and sometimes
better results, up to 26% better.

skantze_replica.py: Tensorflow code that builds and trains a general turn-taking model exactly as described by Skantze in his paper 'Towards a General, Continuous Model of Turn-taking in Spoken Dialogue using LSTM Recurrent Neural Networks' (http://www.sigdial.org/workshops/conference18/proceedings/pdf/SIGDIAL27.pdf)

improved_model.py: Tensorflow code that builds and traing a general turn-taking model with the improves described above

data_reader.py: Code that reads the data used to train either of the two above-mentioned models

