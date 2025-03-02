# DOWNLOADS AND IMPORTS

# Import required packages and functions
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Configure plot style
%matplotlib inline

# Import names
names_present = open('names-present.txt', 'r').read().splitlines()
names_past = open('names-past.txt', 'r').read().splitlines()

#########

# FUNCTION DEFINITIONS

# Function to build the vocabulary of characters and mappings to/from integers
# based on the characters in names present and past.
def build_vocabulary():
  names = names_present + names_past
  chars = sorted(list(set(''.join(names))))
  char_to_int = { char : index + 1 for index, char in enumerate(chars) }
  char_to_int['.'] = 0
  int_to_char = { index : char for char, index in char_to_int.items() }

# build the dataset
def build_dataset(words):
  context_length_in_chars = 3

  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))



########

# SCRIPT LOGIC
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
