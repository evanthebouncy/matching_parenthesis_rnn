# Simple Usages of RNN from TF

## TLDR:

To execute just run 
    
    python matching.py 

## Motivation

An exercise to get the simplest LSTM RNN working from the TensorFlow library, that uses the following features:

1. Construction of a model that contains LSTM units
2. Training of the model
3. Evaluation of the model

Everything is bare-bone minimum as I'm just trying to learn the TF library

## Problem

We're solving the matching parenthesis problem, given a sequence of parenthesis as input, we want to answer if these parenthesis are properly matched on the output.

i.e. 

    ( ( ) ( ) ) -> True
    ( ( ) -> False

## Data Generation

look in data_gen.py for data generation, in particular, we use a 1-hot encoding
to represent ( and ), the sequence is also 0 padded, because TF does not
support dynamic unrolling, to take into account of various different lengths. 

For example, if we have an unrolling of size 6, and our sequence is ( ) ( ), we'll get the following encoding 
    
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

Here, [0, 1, 0] denotes (, [0, 0, 1] denotes ), and [1, 0, 0] denotes the 0 padding

The output is also 1-hot encoded, [1, 0] denotes true, and [0, 1] denotes false

Best way is to load data_gen into the interactive python shell via
execfile('data_gen.py') and execute some of the functions, they're very self
explainatory.

## RNN Model

The RNN model is a simple one, it is unrolled a fixed number of times, say 6, and conceptually perform the following program:

    def matching_paren(sequence):
      state = init_state
      for i in range(0, 6):
        output, state = lstm(sequence[i], state)
      return post_proccess(state)

Here, init_state is the initial state for the lstm unit, which is a vector that
can be learned, lstm is a rnn unit from the tf.models.rnn module, and
post_proccess is a small neural network with a layer of relu and a soft-max
layer for outputing true/false. 

Thus, all the tunable parameters are:

1. The initial state
2. All the weights in the LSTM unit
3. All the weights in the post_process units

See matching.py for details.

## RNN training and evaluation

For training, I compute the softmax as the predicted label. The error is the
cross entropy between the prediction and the true label. For training I'm using
gradient clipping as the gradient can become NaN otherwise, and I'm using
AdaptiveGradient because why the f not (maybe other is better but I have not
gotten around to learn to use those).

For evaluation, I evaluate the performance once every 100 epochs.

## Results

The results are not very good, so far I can only get it to work with
parenthesis up to length of 6, and I believe it is just memorizing all possible
combinations up to length of 6 rather than learning something more general.

I have some hypothesis why the results are bad:

1. The reward structure only reward the RNN at the end of the computation via
true / false. However at each intermediate step we can also detect if the
sequence prefix is correct, this is not currently being expressed in the model.
2. Maybe a stacked and deep RNN would perform better than a flat one

## Remarks
Hope this is helpful! I'm still new to TF so I probably can't answer any questions on TF reliably. Direct all questions to the TF discussion group on google.

