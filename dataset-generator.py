import pandas as pd

# Only keep names comprised of Latin alphabetical characters.
# Keep spaces so model can handle compound names, e.g. Mohamed Salaheldin.
# If name has a space, check that the first word is not a single character,
# assuming names must be 2 or more characters long
def clean_first_name(first_name):
  if not isinstance(first_name, str) or not first_name.isascii() or not first_name.isalpha(): return None
  first_first_name = first_name.split(' ')[0].strip()
  if len(first_first_name) >= 2:
    return first_name.strip()
  else:
    return None

# Read CSV of Egyptian names pulled from Facebook into dataframe
colnames=['first_name', 'last_name', 'gender', 'country']
df = pd.read_csv('fb-eg.csv', names=colnames, usecols=['first_name'])

# Clean, dropping invalid rows replaced with None
df['first_name'] = df.apply(lambda row: clean_first_name(row['first_name']), axis=1)
df.dropna()

# Get unique names
unique_names = df['first_name'].unique()

# Write to text file
with open('names-present-big.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in unique_names))
