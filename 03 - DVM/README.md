# Project 3

## Prerequisites

This project runs on python3

Install python-tk

    apt-get install python3-tk

This project uses a git submodule. Therefore, this project should be cloned with:

    git clone --recurse-submodules https://github.com/pas/pattern-recognition-2018-purple.git

If you already cloned the project, proceed as follows:
1. Move into the folder 'data' inside the '03 - DVM' folder (in Linux) with: `cd 03\ -\ DVM/data/`


2. Then use the following two git commands:

    ```
    git submodule init
    git submodule update
    ```

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
