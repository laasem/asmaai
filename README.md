# Asmaai

Asmaai is a simple 2-layer neural network that generates Egyptian first names.

## Dependencies
Before running, the following dependencies need to be satisfied:

- python3 needs to be installed on the environment you will run the script in (local, cloud, etc.) via command `python3`.

## Run
Assuming python3 is installed via command `python3`, navigate to current directory and run:

```
source .env/bin/activate
python3 script.py
```

to run script in a virtual environment. Please note that the script takes up time and RAM - running on a cloud environment like Google Colab with a GPU runtime would be helpful.

When done, run `source .env/bin/deactivate` to deactivate virtual environment.

## Regenerating dataset
The file `names-present-big.txt` represents the dataset used for training, development, and test. To regenerate it:
1. Download the name dataset from https://github.com/philipperemy/name-dataset.
2. Unzip the folder.
3. Move the `EG.csv` file to the current directory.

## Naming
Asmaai is a portmanteau of the Arabic word "أسماء" ("_asmaa_" or names) and AI.

## Credits
- Egyptian name data pulled from https://github.com/philipperemy/name-dataset.
- Code adapted from https://github.com/karpathy/makemore.
- Developed as part of the NLP course of the Universitat Pompeu Fabra's Master in Theoretical and Applied Linguistics taught by Professor Thomas Brochhagen.
