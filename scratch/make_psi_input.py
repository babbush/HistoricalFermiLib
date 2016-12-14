#Script to make PSI4 Input from Template

import sys
import re

#Get XYZ File
xyz_filename = sys.argv[1]
basename = re.sub('.*/', '', xyz_filename)
basename = re.sub('\.xyz', '', basename) 

#Assume psi4_template is only template file
template_filename = "psi4_template"

#Assume sto-6g basis for now
basis_name = 'sto-3g'
#basis_name = "6-31g"
#basis_name = 'cc-pVDZ'

basename += '_{}'.format(basis_name)

xyz_file = [line for line in open(xyz_filename) if len(line.split()) > 2]
template_file = [line for line in open(template_filename)]

#Add special exception for atoms with high spin
periodic_polarization = [1, 0,
                         1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
                         1, 0, 1, 2, 5, 6, 5, 8, 9, 0, 1, 0, 1, 2, 3, 2, 1 ,0]
element_list = ['X', 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al', 'Si','P','S','Cl','Ar','K','Ca','Sc', 'Ti','V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Ha']

if (len(xyz_file) == 1):
    atom_name = xyz_file[0].split()[0]
    atom_number = element_list.index(atom_name)
    atom_spin_state = periodic_polarization[atom_number-1] + 1
    xyz_file.append('\n 0 {}'.format(atom_spin_state))

input_file = [re.sub('#GEOMETRY#', ''.join(xyz_file), line) 
              for line in template_file]
input_file = [re.sub('#BASE_NAME#', basename, line) 
              for line in input_file]
input_file = [re.sub('#BASIS#', basis_name, line) 
              for line in input_file]

with open(basename+'.inp', 'w') as output_file:
    output_file.write(''.join(input_file))

  

