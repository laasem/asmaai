# DOWNLOADS AND IMPORTS

# Download pip and use it to install required packages
import pip
required_packages = ['torch', 'matplotlib']
for package in required_packages:
  print(f'Installing package: {package}...')
  pip.main(['install', package])

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

# HYPERPARAMETERS
CONTEXT_LENGTH_IN_CHARS = 3
NUM_CHAR_EMBEDDING_DIMENSIONS = 10

#########

# FUNCTION DEFINITIONS

# Function to build vocabulary of words by concatenating
# names present and past
def vocabulary():
  return names_present

# Function that creates a dict mapping characters in vocabulary
# to integers as a way of encoding them
def encoder():
  chars = sorted(list(set(''.join(vocabulary()))))
  char_to_int = { char : index + 1 for index, char in enumerate(chars) }
  char_to_int['.'] = 0
  return char_to_int

# Function that creates a dict mapping integers back to
# characters in vocabulary to decode them
def decoder(encoder):
  return { index : char for char, index in encoder.items() }

# Function to build dataset of inputs and gold labels
# based on given vocabulary (i.e. set of names)
# and context length in chars (i.e. how many chars are we using to predict the next chars)
def build_dataset(data, encoder):
  # X represents inputs and Y gold labels
  X, Y = [], []

  # Loop over each given vocabulary item (i.e. name)
  for name in data:

    # Create an array of 0s the size of the desired context
    # So if desired context is 3 chars, array will be [0, 0, 0]
    context = [0] * CONTEXT_LENGTH_IN_CHARS

    # Loop over each char in name, adding a period to serve as an end token
    for char in name + '.':

      # Encode char
      integer = encoder[char]

      # Add current context to input
      X.append(context)

      # Add encoded char/integer to gold label
      Y.append(integer)

      # Update context with encoded char
      context = context[1:] + [integer]

  # Convert inputs and gold labels to tensors
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

# Function to initialize parameters
# i.e. vector of character embeddings + weights and biases
def initialize_parameters(encoder):
  # Initialize PyTorch generator with constant manual seed for reproducibility
  generator = torch.Generator().manual_seed(2147483647)

  # Build vector of character embeddings
  num_chars = len(encoder.keys())
  C = torch.randn((num_chars, NUM_CHAR_EMBEDDING_DIMENSIONS), generator=generator)

  # Build 2 sets of weights and biases for a neural network of 2 layers
  w1_num_rows = CONTEXT_LENGTH_IN_CHARS * NUM_CHAR_EMBEDDING_DIMENSIONS
  w1_num_cols = 200
  W1 = torch.randn((w1_num_rows, w1_num_cols), generator=generator)
  b1 = torch.randn(w1_num_cols, generator=generator)


  w2_num_rows = w1_num_cols
  w2_num_cols = num_chars
  W2 = torch.randn((w2_num_rows, w2_num_cols), generator=generator)
  b2 = torch.randn(w2_num_cols, generator=generator)

  parameters = [C, W1, b1, W2, b2]

  # Initialize gradients in parameter tensors
  for parameter in parameters: parameter.requires_grad = True

  return parameters

def train_model(parameters, Xtrain, Ytrain):
  [C, W1, b1, W2, b2] = parameters
  stepi = []
  lossi = []

  for i in range(200000):
    # Construct minibatch of size 32
    index = torch.randint(0, Xtrain.shape[0], (32,))

    # FORWARD PASS

    # HIDDEN LAYER

    # Reshape character embedding to same shape as W1 and b1
    # to be able to multiply to multiply them
    char_embedding = C[Xtrain[index]].view(-1, 30)

    # Multiply character embedding by weight and bias
    z = char_embedding @ W1 + b1

    # Use tanh activation function for non-linearity
    h = torch.tanh(z)

    # OUTPUT LAYER

    # Multiply output of hidden layer by weight and bias of second/output layer
    # to get logits
    logits = h @ W2 + b2

    # Calculate the loss by comparing the logits and gold label distributions
    loss = F.cross_entropy(logits, Ytrain[index])

    # BACKWARD PASS
    for parameter in parameters: parameter.grad = None
    loss.backward()

    # Initialize learning rate, starting out faster then slowing down
    lr = 0.5 if i < 100000 else 0.05

    # Update gradients
    for parameter in parameters: parameter.data += -lr * parameter.grad

    # Track steps and loss at each step
    stepi.append(i)
    lossi.append(loss.log10().item())

  return [parameters, stepi, lossi]

########

# SCRIPT LOGIC

# 'Randomly' shuffle data, using constant seed for reproducibility
random.seed(42)
data = vocabulary()
random.shuffle(data)

# Create train, dev, and test dataset splits, where
# 80% of data is used for training,
# 10% for development/validation, and
# 10% for testing
split_point1 = int(0.8 * len(data))
split_point2 = int(0.9 * len(data))

encoder = encoder()
decoder = decoder(encoder)

Xtrain, Ytrain = build_dataset(data[:split_point1], encoder)
Xdev, Ydev = build_dataset(data[split_point1:split_point2], encoder)
Xtest, Ytest = build_dataset(data[split_point2:], encoder)

# Train model, getting learned parameters and calculating loss
parameters = initialize_parameters(encoder)
[learned_parameters, step, loss] = train_model(parameters, Xtrain, Ytrain)

# Plot loss
plt.plot(step, loss)

# Run model to generate names
NUM_OF_NAMES_TO_GENERATE = 100

[C, W1, b1, W2, b2] = learned_parameters
generator = torch.Generator().manual_seed(299283892)

for _ in range(NUM_OF_NAMES_TO_GENERATE):
    context = [0] * CONTEXT_LENGTH_IN_CHARS
    out = []

    while True:
      char_embedding = C[torch.tensor([context])]
      h = torch.tanh(char_embedding.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      integer = torch.multinomial(probs, num_samples=1, generator=generator).item()
      context = context[1:] + [integer]
      out.append(integer)
      if integer == 0: break

    print(''.join(decoder[i] for i in out))

# training loss
emb = C[Xtrain] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytrain)
print(loss)

# validation loss
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
print(loss)

# test loss
emb = C[Xtest] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytest)
print(loss)
