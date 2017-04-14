#Script to merge CC amplitudes from the output file into the pickle file

import sys
sys.path.append('..')
import re
import cPickle as pickle
import MolData

def merge_amplitudes(output_filename, pickle_name):
    #Parse the output file for amplitudes
    output_buffer = [line for line in open(output_filename)]

    T1IA_index = 0
    T1ia_index = 0
    T2IJAB_index = 0
    T2ijab_index = 0
    T2IjAb_index = 0

    #Find Start Indices
    for i, line in enumerate(output_buffer):
        if ("Largest TIA Amplitudes:" in line):
            T1IA_index = i

        elif ("Largest Tia Amplitudes:" in line):
            T1ia_index = i

        elif ("Largest TIJAB Amplitudes:" in line):
            T2IJAB_index = i

        elif ("Largest Tijab Amplitudes:" in line):
            T2ijab_index = i

        elif ("Largest TIjAb Amplitudes:" in line):
            T2IjAb_index = i

    T1IA_Amps = []
    T1ia_Amps = []

    T2IJAB_Amps = []
    T2ijab_Amps = []
    T2IjAb_Amps = []

    #Read T1's
    if (T1IA_index is not 0):
        for line in output_buffer[T1IA_index+1:]:
            ivals = line.split()
            if not ivals: break
            T1IA_Amps.append([int(ivals[0]), int(ivals[1]), float(ivals[2])])

    if (T1ia_index is not 0):
        for line in output_buffer[T1ia_index+1:]:
            ivals = line.split()
            if not ivals: break
            T1ia_Amps.append([int(ivals[0]), int(ivals[1]), float(ivals[2])])

    #Read T2's
    if (T2IJAB_index is not 0):
        for line in output_buffer[T2IJAB_index+1:]:
            ivals = line.split()
            if not ivals: break
            T2IJAB_Amps.append([int(ivals[0]), int(ivals[1]), 
                                int(ivals[2]), int(ivals[3]),
                                float(ivals[4])])

    if (T2ijab_index is not 0):
        for line in output_buffer[T2ijab_index+1:]:
            ivals = line.split()
            if not ivals: break
            T2ijab_Amps.append([int(ivals[0]), int(ivals[1]), 
                                int(ivals[2]), int(ivals[3]),
                                float(ivals[4])])

    if (T2IjAb_index is not 0):
        for line in output_buffer[T2IjAb_index+1:]:
            ivals = line.split()
            if not ivals: break
            T2IjAb_Amps.append([int(ivals[0]), int(ivals[1]), 
                                int(ivals[2]), int(ivals[3]),
                                float(ivals[4])])

    #Open the pickle and insert the data
    molecule = pickle.load( open(pickle_name, 'rb') )
    molecule.T1IA_ = T1IA_Amps
    molecule.T1ia_ = T1ia_Amps
    molecule.T2IJAB_ = T2IJAB_Amps
    molecule.T2ijab_ = T2ijab_Amps
    molecule.T2IjAb_ = T2IjAb_Amps

    #Dump it back to the original file
    pickle.dump( molecule, open(pickle_name, 'wb' ) )

if __name__=="__main__":
    output_filename = sys.argv[1]
    base_name = re.sub('\.out', '', output_filename)
    pickle_name = base_name + '.pkl'

    merge_amplitudes(output_filename, pickle_name)
