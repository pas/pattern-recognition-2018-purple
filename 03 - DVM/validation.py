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
        if img.split("-")[0] == 300:
            return line
        else:
            line = line + 1
    return line

# Returns true if line in transcription file represents valid image.
def check_if_valid_line(valid_img):
    if (int(valid_img.split("-")[0]) > 300):
        return True
    else:
        return False



# TODO: Main part here. Probably we have to give this part a class name so that it can be called from other classes.
# TODO: Also refactor: make more methods.
keywords = open("data/PatRec17_KWS_Data/task/keywords.txt", "r")

# Do validation for each keyword.
for keyword in keywords:
    keyword = keyword.split("\n")[0] # We do not want to have the linebreak included in the string.
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
        if content == keyword:
            train_images.append([ jpg, counter])

    # Part B: For each word image of the validation set calculate DTW value and store it together with the actual content of the imageself.
    valid_images = open("data/PatRec17_KWS_Data/ground-truth/transcription.txt", "r")
    counter = 0
    jpg = 300
    for train_img in train_images:
        # TODO: open train_img
        DTW_values = [] # Stores DTW values of train_img and each image of valid set. Also stores the word of the valid image.
        for valid_img in valid_images:
            if check_if_valid_line(valid_img): # Only lines which store information about image of valid set are relevant.
                counter = counter + 1
                actual_jpg = valid_img.split("-")[0]
                if jpg != int(actual_jpg):
                    jpg = int(actual_jpg)
                    counter = 1
                content_of_img = valid_img.split()[1]
                path_to_img = "images/"+ str(jpg) +"/image-" + str(counter)
                # TODO: open valid image and calculate DTW for this valid img and the train img.
                # TODO: Store DTW together with content_of_img in an DTW_values array.
        #TODO:Sort that DTW_array and calculate plots for precision and recall.
