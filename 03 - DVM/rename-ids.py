from paths import Paths

paths = Paths()

content = None
with open("validation-output.txt") as f:
    content = f.readlines()

for page_number in range(305, 310):
  _, ids = paths.get( "validation/ground-truth/locations/" + str(page_number) + ".svg" )
  
  for counter,id in enumerate(ids):
    for index, line in enumerate(content):
      content[index] = line.replace(str(page_number) + "/image-"+str(counter+1)+",", str(id)+",")

# Write file
with open("validation-output-ids.txt", "w") as f:  
  for line in content:
    f.write( line )

      
    
  

