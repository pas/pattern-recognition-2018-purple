# Project 3

## VALIDATION - LAST TASK

runner-validation.py can be used to start the validation.

The data is appended to validation-output.txt

### Changes
Some changes were necesseary to adapt our program to output the correct format. This lead to the creation of a new runner called runner-validation.py and changes especially for validation. Does are cleary marked by all the if statements checking for is_validation.

During the validation we recognized a mature flaw in our code. When constructing the feature vectors the image was not scanned from left to right but from top to bottom. As well we missed out to normalize the features. A small change in the image.py file was done to correct the issue of wrong scanning. As well the features were normalize in image.py. We adapted as well the sakoe-chiba bandwith.

### Quick fix for ID problem
This was already done for the current data so the validation-output.txt holds the correct ids! 

The ids are wrongly named after the validation-output.txt is created. If you recreate the data you can use rename-ids.py to create a new file called validation-output-ids.txt with the correct ids. 

## Prerequisites

This project runs on python3

Install python-tk

    apt-get install python3-tk

This project uses a git submodule. Therefore, this project should be cloned with:

    git clone --recurse-submodules https://github.com/pas/pattern-recognition-2018-purple.git

If you already cloned the project, proceed as follows:
1. Move into the folder 'data' inside the '03 - DVM' folder (in Linux) with: `cd 03\ -\ DVM/data/`


2. Then use the following two git commands:

    git submodule init
    git submodule update

It is best to use a virtual environment. Then install requirements with:

    pip install -r requirements.txt

If an IDE is used, the following line can be used to generate a pylintrc file:

    pylint --extension-pkg-whitelist=cv2,PIL --indent-string='  ' --disable=C  --generate-rcfile > .pylintrc

## Directory

The following files are provided

    runner.py # The main program. Dies the preprocessing, testing and validation.

    test.py # Testing classes and methods

    image.py, paths.py # Helper files for preprocessing.
    validation.py, metrics.py, dtw.py, feature.py # Helper file for testing and validation

The directory is constructed the following way

    |- tests # Testing the different classes
    |- data # PatRec17_KWS_Data
    |- images # Preprocessed images
    |- plots # Precision recall plots generated through validation
    |- validation # All data for validation

## Usage

runner.py is the main program.
First, it does preprocess the given jpg files.
It splits the jpg images into its words. For each jpg file a subfolder in images is created and the words of this jpg are stored in this folder.
E.g. all words of 270.jpg are stored in images/270, named in increasing order (image-1, image-2 etc.) of their appearance in the jpg file.
Second, it  does the training and validation.
For each keyword the related train images is fetched. This train image is then compared with every validation image: The DTW value and also a boolean which states if the train and validation image represent the same word are stored in a DTW_values array.  After having compared every validation image with the train image, the DTW_values array gets sorted and the recall precision plot gets created.

## Results and Interpretation

You can find our precision recall plots in the plots folder after having run the main program (runner.py). For each keyword there is a separate folder. For all train images which represent the keyword, validation is done and a plot gets calculated.

For the plots, we would expect that the precision decreases with an increasing recall. This would give a falling curve. Since our sorted DTW vectors do not contain the positives first and the negatives in the end, we can not observe this falling curve in our plots.
