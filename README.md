# Simple Usages of RNN from TF

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

look in data_gen.py for data generation, in particular, we use a 1-hot encoding to represent ( and ), the sequence is also 0 padded, because TF does not support dynamic unrolling, to take into account of various different lengths. 

For example, if we have an unrolling of size 6, and our sequence is ( ) ( ), we'll get the following encoding 
    
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

Here, [0, 1, 0] denotes (, [0, 0, 1] denotes ), and [1, 0, 0] denotes the 0 padding

The output is also 1-hot encoded, [1, 0] denotes true, and [0, 1] denotes false

## RNN Model

The RNN model is a simple one, it is unrolled a fixed number of times, say 6, and conceptually perform the following program:

    def matching_paren(sequence):
      state = init_all_zero()
      for i in range(0, 6):
        output, state = lstm(sequence[i], state)
      return post_proccess(state)

Here, lstm is a rnn unit from the tf.models.rnn module, and post_proccess is a neural network unit with a layer of relu and a soft-max layer. See matching.py for details

## Results

The results are not very good, so far I can only get it to work with parenthesis up to length of 6, and I believe it is just memorizing all possible combinations up to length of 6 rather than learning something more general.

I have some hypothesis why the results are bad:

1. The initial state is all 0, whereas for an interesting recurrence, the initial state should be a variable that can be learned jointly
2. The reward structure only reward the RNN at the end of the computation via true / false. However at each intermediate step we can also detect if the sequence prefix is correct
3. Maybe a stacked and deep RNN would perform better than a flat one


  
