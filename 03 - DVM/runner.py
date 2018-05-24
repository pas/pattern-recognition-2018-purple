#
# Images are stored into images
#

import Image
import os

from image import ImageProcessor

from preprocess import Preprocessor
from paths import Paths
from validation import Validation


# part 1: Image preprocessing.
pathToProvidedData = "data/PatRec17_KWS_Data/"
paths = Paths()

preprocess = Preprocessor()

images = []

# iterate through all the page images (jpg files) and through the related svg files
for page in os.listdir(pathToProvidedData + "images"):
    page_number = page[:-4] #extracts number from filename; so 240.jpg gets 240
    print(page)
    imageP = ImageProcessor(pathToProvidedData + "images/" + str(page_number) + ".jpg")
    polygons = paths.get(pathToProvidedData + "ground-truth/locations/" + str(page_number) + ".svg")

    # word of each page image (jpg file) are put in a seperate folder
    os.makedirs("images/" + str(page_number))
    number = 1
    for path in polygons:
        #print (path)
        new_image = imageP.crop(path)
        new_image = preprocess.binarization( new_image )
        new_image = ImageProcessor.trim_white_space_on_array( new_image )
        images.append( new_image )
        new_image = Image.fromarray( new_image )
        new_image = ImageProcessor.resize_image( new_image, 200 )
        new_image.save("images/"+ str(page_number) + "/image-"+str(number) +'.png')
        number += 1

# part 2: Training and validation.
Validation.do_validation( pathToProvidedData );
