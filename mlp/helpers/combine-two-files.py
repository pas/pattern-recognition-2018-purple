file1 = "data/train.csv"
file2 = "data/test.csv"
new_file = open("data/combined.csv", 'w+')

with open(file1, 'rb') as test_file:  
  for line in test_file:
    new_file.write( line )

with open(file2, 'rb') as test_file:  
  for line in test_file:
    new_file.write( line )

new_file.close()
