from image import ImageProcessor
from dtw import DTW
from metrics import Metrics

#
# Validation.
# For each keyword fetch the related train images.
# Then compare each of these images with all the validation images and calculate DTW.
# Sort the DTW's in increasing order and calculate precision and recall.
#

# Returns the line where the translation of the valid images in the transcription file starts.
def get_start_of_valid():
    line = 0
    images = open("data/PatRec17_KWS_Data/ground-truth/transcription.txt", "r")
    for img in images:
        # Looking for first occurence of 300 because
        # the image 300 is the start of our validation
        # set
        if img.split("-")[0] == 300:
            return line
        else:
            line = line + 1
    return line

# Returns true if line in transcription file represents valid image.
# All images in the valdation set are in the files that are
# greater than 300
def check_if_valid_line(valid_img):
    return int(valid_img.split("-")[0]) > 300

def get_key( item ):
  return item[0]
  
# TODO: Main part here. Probably we have to give this part a class name so that it can be called from other classes.
# TODO: Also refactor: make more methods.
keywords = open("data/PatRec17_KWS_Data/task/keywords.txt", "r")

# Do validation for each keyword.
for keyword in keywords:
    keyword = keyword.split("\n")[0] # We do not want to have the linebreak included in the string.
    print( "Keyword: " + keyword )
    train_images = [] # Stores test images of the keyword (jpg file and number of word in jpg file).

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
            if jpg > 279: # Only the jpg files 270-279 belong to the train set.
                break
            jpg = int(actual_jpg)
            counter = 1
        if content.startswith( keyword ):
            path_to_img = "images/"+ str(jpg) +"/image-" + str(counter) + ".png"
            train_images.append( ImageProcessor( path_to_img ) )
            print( "Found training images: " + "images/"+ str(jpg) +"/image-" + str(counter) + ".png" )

    # Part B: For each word image of the validation set calculate DTW value and store it together with the actual content of the imageself.
    dtw = DTW()
    
    for train_img in train_images:
        valid_images = open("data/PatRec17_KWS_Data/ground-truth/transcription.txt", "r")
        counter = 0
        jpg = 300
      
        # TODO: open train_img
        DTW_values = [] # Stores DTW values of train_img and each image of valid set. Also stores the word of the valid image.
        train_features = train_img.calculate_feature_vectors()
        
        for valid_img in valid_images:
            if check_if_valid_line(valid_img): # Only lines which store information about image of valid set are relevant.
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
                
                # Store DTW (? together with content_of_img ?) in an DTW_values array.
                # TODO: find out if image fits keyword
                dist, _ = dtw.distance( train_features , validation_features , 4 )
                validation_string = valid_img.split(" ")[1][:-1]
                if( validation_string.startswith( keyword ) ):
                  print( "Found same text with distance " + str(dist) + ": " + path_to_img )
                DTW_values.append( [ dist , keyword , validation_string , validation_string.startswith( keyword ) ] )
                
            DTW_values = sorted( DTW_values , key=get_key )
            Metrics.plot_recall_precision( DTW_values )
        #TODO:Sort that DTW_array and calculate plots for precision and recall.
