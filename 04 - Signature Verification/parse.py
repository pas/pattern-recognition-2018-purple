##
#
# Reads a folder of signatures txt input files and stores them in a hash.
#
##
import os

class Parser:

    def read(folder):
        signatures = {}
        for file in os.listdir(folder):
            signature_label = file
            signature_properties = []
            for line in open(folder + file):
                properties_at_timepoint = [ prop.strip() for prop in line.split(' ')]
                signature_properties.append(properties_at_timepoint)
            signatures[signature_label] = signature_properties
        return signatures
