import csv
import random
import os, shutil

class Fold: 
  fold_number = 0
  fold_file = ""
  
  def __init__(self, fold_number):
      self.fold_number = fold_number
      self.fold_file = open('folds/fold-'+str(fold_number)+'.csv', 'w+')

# Empty folds folder
folder = 'folds'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
    except Exception as e:
        print(e)

folds = []
for x in range(0, 4):
  folds.append( Fold(x) )

with open('data/train-orig.csv', 'rb') as test_file:
  numbers = []
  
  for line in test_file:
    fold_number = 0;
    if( len(numbers) != 0 ):
      fold_number = numbers.pop();
    else:
      numbers = [ 0 , 1 , 2 , 3 ]
      random.shuffle( numbers )
      fold_number = numbers.pop()
    # Add line to selected file
    folds[ fold_number ].fold_file.write( line )

for fold in folds:
  fold.fold_file.close()
