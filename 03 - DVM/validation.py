from image import ImageProcessor
from dtw import DTW
from metrics import Metrics
import os
import numpy as np

#
# Validation.
# For each keyword fetch the related train images.
# Then compare each of these images with all the validation images and calculate DTW.
# Sort the DTW's in increasing order and calculate precision and recall.
#

class Validation:

    # Returns true if line in transcription file represents valid image.
    # All images in the valdation set are in the files that are
    # greater than or equal 300
    def check_if_valid_line(self, valid_img):
        return int(valid_img.split("-")[0]) >= 300

    def get_key( self, item ):
      return item[0]

    def load_all_validation_images(self):
        validation_images = []
        valid_images = open("data/PatRec17_KWS_Data/ground-truth/transcription.txt", "r")
        counter = 0
        jpg = 300

        for valid_img in valid_images:
            if Validation().check_if_valid_line(valid_img): # Only lines which store information about image of valid set are relevant.
                counter = counter + 1
                actual_jpg = valid_img.split("-")[0]
                if jpg != int(actual_jpg):
                    jpg = int(actual_jpg)
                    counter = 1
                content_of_img = valid_img.split()[1]
                path_to_img = "images/"+ str(jpg) +"/image-" + str(counter) + ".png"

                # open valid image and calculate DTW for this valid img and the train img.
                image = ImageProcessor( path_to_img )
                validation_features = image.calculate_feature_vectors()
                
                validation_string = valid_img.split(" ")[1][:-1]

                validation_images.append( [ validation_features , validation_string ] )
        return validation_images
    
    # Added for validation
    def load_validation_images(self):
        validation_images = []
        counter = 0
        jpg = 305
        
        for jpg in range(305, 310):
          directory = "images/" + str(jpg)
          counter = 0
          if( os.path.exists( directory ) ):
            for filename in os.listdir( directory ):
              counter = counter + 1
              path_to_img = "images/"+ str(jpg) +"/image-" + str(counter) + ".png"
              image = ImageProcessor( path_to_img )
              validation_features = image.calculate_feature_vectors()
              # ID is constructed out of the file number and the image nr
              validation_string = str(jpg) +"/image-" + str(counter)
              validation_images.append( [ validation_features , validation_string ] )
          else:
            print( "ERROR: Missing data for images/" + str(page_number) )

        return validation_images

    @staticmethod
    def do_validation( pathToProvidedData , is_validation=False ):

        if( is_validation ):
          validation_img = Validation().load_validation_images()
        else:
          validation_img = Validation().load_all_validation_images()

        keywords = open(pathToProvidedData + "/task/keywords-1.txt", "r")

        # Do validation for each keyword.
        for keyword in keywords:
            keyword = keyword.split("\n")[0] # We do not want to have the linebreak included in the string.
            
            #only used for validation run
            keyword_location = ""
            # 
            # The validation set provides us with the exact
            # location of the word which we simply cut away
            #
            if( is_validation ):
              keyword, keyword_location = keyword.split(",")
              print( "Word location: " + keyword_location )
            
            print( "Keyword: " + keyword )
            train_images = [] # Stores train images of the keyword (jpg file and number of word in jpg file).

            # Part A: Store location of train images which contain the keyword.
            images = open("data/PatRec17_KWS_Data/ground-truth/transcription.txt", "r")
            jpg = 270 # Stores the number of the jpg file the test-image comes from.
            counter = 0 # Counts up the number of the word in the jpg file. Is needed to read the actual png (are stored with decreasing numbers).
            for img in images:
                counter = counter + 1
                location = img.split()[0]
                content = img.split()[1]
                actual_jpg = location.split("-")[0]
                if jpg != int(actual_jpg):
                    if int(actual_jpg) > 279 and int(actual_jpg) != 300: # Only the jpg files 270-279 and 300 belong to the train set.
                        continue
                    jpg = int(actual_jpg)
                    counter = 1
                if ( ( content.startswith( keyword ) and not is_validation ) or ( content.startswith( keyword ) and is_validation and keyword_location == location )  ):
                    path_to_img = "images/"+ str(jpg) +"/image-" + str(counter) + ".png"
                    train_images.append( ImageProcessor( path_to_img ) )
                    print( "Found training images: " + "images/"+ str(jpg) +"/image-" + str(counter) + ".png" )

            # Part B: For each word image of the validation set calculate DTW value and store it together with the actual content of the imageself.
            dtw = DTW()

            for train_img in train_images:
                print( "Compare to " + train_img.image_name )

                DTW_values = [] # Stores DTW values of train_img and each image of valid set. Also stores the word of the valid image.
                train_features = train_img.calculate_feature_vectors()

                for validation_features , validation_string in validation_img:
                    # Widened the sakoe-chiba-bandwith
                    dist, _ = dtw.distance( train_features , validation_features , 10 )
                    if( validation_string.startswith( keyword ) ):
                      print( "Found same text with distance " + str(dist) + ": " + path_to_img )
                    DTW_values.append( [ dist , keyword , validation_string , validation_string.startswith( keyword ) ] )

                DTW_values = sorted( DTW_values , key=Validation().get_key )
                keyword_valid_match = [ a[3] for a in DTW_values ] # Takes last column for each array in DTW_values.
                
                
                # The expected outcome for the validation is: 
                # Keyword1, testword_ID1, dissimilarity1, testword_ID2, dissimilarity2, ...
                # Keyword2, testword_ID1, dissimilarity1, testword_ID2, dissimilarity2, ...
                # The result is put into the validation-output.txt file
                if( is_validation ):
                  # Open file for appending
                  f=open("validation-output.txt",'ab')
                  # Remove unnecessary columns
                  DTW_values = np.array( DTW_values )
                  DTW_values = np.delete(DTW_values, [1,3], axis=1)
                  # Swap columns
                  DTW_values[:,[1,0]] = DTW_values[:,[0,1]]
                  # Flatten the array from [ [ v11 , v12 ] , [ v21 , v22] ] to [ v11, v12, v21, v22 ]
                  DTW_values = DTW_values.flatten()
                  # Add keyword at the beginning
                  DTW_values = np.insert(DTW_values, 0, keyword)
                  print ( DTW_values )
                  # We have to force the DTW_value to a 2D-array because otherwise it
                  # will output every element in the array as one line
                  np.savetxt(f, np.atleast_2d(DTW_values), delimiter=",", fmt="%s")
                  # Close file
                  f.close()
                else:
                  Metrics.plot_recall_precision( keyword_valid_match, keyword, train_img.image_name )
                
