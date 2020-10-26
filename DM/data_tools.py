import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def find_all(s, ch):
  """Returns all occurrences of character ch in string s"""
  return [i for i, ltr in enumerate(s) if ltr == ch]

def seq2matrix(sequence):
  """Encodes the passed fasta sequence into an incidence matrix, seq_length x 4"""

  #declare the result array
  res = np.zeros((len(sequence), 4))

  #for each base, translate it to the corresponding encoding
  pos = find_all(sequence.upper(), 'A')
  res[pos, :] = [1, 0, 0, 0]

  pos = find_all(sequence.upper(), 'C')
  res[pos, :] = [0, 1, 0, 0]

  pos = find_all(sequence.upper(), 'G')
  res[pos, :] = [0, 0, 1, 0]

  pos = find_all(sequence.upper(), 'T')
  res[pos, :] = [0, 0, 0, 1]

  #and we are done
  return(res)

def pad_fasta(fasta, ncols):
  """Adds zero padding to the passed fasta (incidence matrix) to reach desider number of columns ncols"""
  for i in range(len(fasta)):
    fasta[i] = pad_matrix(fasta[i], ncols)
  return(fasta)

def pad_matrix(matrix, ncols):
  """Adds zero padding to the passed matrix to reach desider number of columns ncols"""
  #error condition
  if matrix.shape[0] > ncols:
    raise ValueError("Requested padding to " + str(ncols) + " columns but input matrix is already bigger")
  
  #if already good no operation is needed
  if matrix.shape[0] == ncols:
    return(matrix)
  
  #if we get here padding is needed
  padding = np.zeros((ncols - matrix.shape[0], 4))
  return(np.vstack((matrix, padding)))

def load_fasta(fasta_filename):
  """Loads the passed fasta into a list of matrices"""
  res = []
  with open(fasta_filename, 'r') as f:
    for line in f:
      #fasta is alternating between title and sequence
      #let's discard titles
      if line.startswith('>'):
        continue
      
      #if we get here is a legit sequence line
      res.append(seq2matrix(line.rstrip()))

  return(res)

def fasta2cube(fasta_list):
  """from list of matrices to 3D numpy array, lines x sequence x 4"""
  lines = len(fasta_list)
  seq = fasta_list[0].shape[0]

  #room for result
  res = np.zeros((lines, seq, 4))

  #first free line
  ffl = 0

  for i in range(len(fasta_list)):
    res[ffl, :, :] = fasta_list[i]
    ffl += 1

  #and we are done
  return(res)

def get_seq_length(loaded_fasta, min_length):
  """Returns the maximum sequence length found in the passed list of fasta matrices, or min_length, if higher"""
  m = min_length
  for i in range(len(loaded_fasta)):
    l = loaded_fasta[i].shape[0]
    if l > m:
      m = l
  return (m)

def build_set(features_dict, classes_sequence):
  """Builds a feature cube (num_reads x sequence_length x 4) and a
  classes array (num_reads, classes labels are repeated in blocks). It
  requires a features dictionary (keys are classes labels, values are 
  arrays of fasta matrices as those returned by load_fasta) and an array of
  classes (class label 1, class label 2, ...) that will be used to force
  classes order in the final cube.  
   """

  #count number of samples and, since we are at it, the sequence length
  sam = 0
  sequence_length = 0
  for cl in features_dict.keys():
    sam += len(features_dict[cl])
    sequence_length = features_dict[cl][0].shape[0]
  
  #building room for the result feature cube
  res = np.zeros((sam, sequence_length, 4))

  #room for the labels
  classes = []

  #first free row
  ffr = 0 

  #filling the result, keeping the order specified
  for cl in classes_sequence:
    #building the cube
    last_line = ffr + len(features_dict[cl])
    res[ffr:last_line, :, :] = fasta2cube(features_dict[cl])
    ffr = last_line
    #building the labels
    classes = np.hstack([classes, [cl] * len(features_dict[cl])])

  #and we are done
  return((res, classes))

def load_all_fastas(data_summary, val_split = None):
  """builds and returns features cube and class labels array for
  train, test and optionally validation sets, with the data specified in the passed 
  data summary pandas dataframe (expected columns: file, class, 
  training). Argument val_split can be either a fraction in [0,1] or None"""

  #we start by loading train/test data

  #file, class, training. 
  train_features = {}
  test_features = {}
  train_class_progression = []
  test_class_progression = []

  max_seq_length = 0

  for i in range(data_summary.shape[0]):
    #extract current line data
    fi = data_summary.iloc[i, 0]
    cl = data_summary.iloc[i, 1]
    tr = data_summary.iloc[i, 2]

    #interface
    print('Loading class: ' + cl + ' [ Train:', tr, ']')

    #read the fasta data
    fa = load_fasta(fi)

    #should we update the current max length?
    max_seq_length = get_seq_length(fa, max_seq_length)

    #store in the appropriate boxes
    if tr:
      train_features[cl] = fa
      train_class_progression.append(cl)
    else:
      test_features[cl] = fa
      test_class_progression.append(cl)
    
  #pad, if necessary
  for cl in train_features.keys():
    train_features[cl] = pad_fasta(train_features[cl], max_seq_length)
  for cl in test_features.keys():
    test_features[cl] = pad_fasta(test_features[cl], max_seq_length)
  
  #load everything in two big matrices, plus classes arrays
  (train_features, train_classes) = build_set(train_features, train_class_progression)
  (test_features, test_classes)   = build_set(test_features,  test_class_progression)
  
  #if we don't have a validation set, we are done
  if val_split is None:
    return(train_features, train_classes, test_features, test_classes)
   
  #if we get here, we need to carve a validation set from the train data
  sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split)
  for train_index, val_index in sss.split(train_features, train_classes):
    val_features = train_features[val_index, :, :]
    val_classes  = train_classes[val_index]
    train_features = train_features[train_index, :, :]
    trian_classes  = train_classes[train_index]
  
  #and we are done
  return(train_features, train_classes, test_features, test_classes, val_features, val_classes)

