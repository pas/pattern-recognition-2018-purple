# Project 3

## Prerequisites

This project uses a git submodule. Therefore, this project should be cloned with:

    git clone --recurse-submodules https://github.com/pas/pattern-recognition-2018-purple.git

If you already cloned the project, proceed as follows:
1. Move into the folder 'data' inside the '03 - DVM' folder (in Linux) with:
    cd 03\ -\ DVM/data/
2. Then use the following two git commands:
    git submodule init
    git submodule update

It is best to use a virtual environment. Then install requirements with:

    pip install -r requirements.txt

If an IDE is used, the following line can be used to generate a pylintrc file:

    pylint --extension-pkg-whitelist=cv2,PIL --indent-string='  ' --disable=C  --generate-rcfile > .pylintrc

## Directory

The following files are provided

    runner.py # The main program
    test.py # Testing classes and methods

The directory is constructed the following way

    |- tests # Testing the different classes
    |- images # Preprocessed images

## Usage

## Results

## Conclusions
