"""
MolData.py - Module for defining molecular data format that is common to quanutum chemical calculations.
"""
import numpy as np
import scipy as sp

__author__ = "Jarrod R. McClean"
__email__ = "jarrod.mcc@gmail.com"

class MolData(object):
    """Molecular data class for storing molecule data from a fixed basis set of size M,
    at a fixed molecular geometry.  This is a convenient common format to store molecular
    data with pickling.  Not every field need be filled in every calculation."""
    def __init__(self):
        """Initialize molecular data and define fields
        M - Number of basis functions defining the molecule"""
        #Scalars needed to define or check the energy or components of the energy
        self.M_ = None #Number of spatial basis functions in the calculation.
        self.charge_ = None #Overall charge the calculations were performed at
        self.nAtoms_ = None #Number of atoms in the molecule
        self.Z_ = None #Array storing nuclear charges for each atom
        self.multiplicity = None #Spin multiplicity of the source calculation
        self.nElectrons_ = None #Total Electrons
        self.nFrozen_ = 0 #Frozen Doubly Occupied Orbitals
        self.nuclearRepulsion_ = None #Nuclear Repulsion constant
        self.SCFEnergy_ = None #RHF Energy
        self.USCFEnergy_ = None #Unrestricted/Spin-Symmetry Broken Energy
        self.MP2Energy_ = None #MP2 Energy
        self.CISDEnergy_ = None #CISD Energy
        self.CCSDEnergy_ = None #CCSD Energy
        self.FCIEnergy_ = None #FCI energy
        self.hasECP_ = False #Flag to determine if calculation was done with effective core potential

        #Basis set information
        self.basisToAtom_ = None #List assigning basis function i to atom number basisToAtom_[i]

        #One electron properties
        self.overlap_ = None #Overlap matrix of basis set
        self.kinetic_ = None #Kinetic energy matrix elements
        self.nuclear_ = None #Nuclear attraction integrals
        self.pseudo_ =  None #Pseudo-potential integrals
        self.totalCore_ = None #Total core one electron integrals (kinetic + nuclear)
        self.totalCoreECP_ = None #Total core with ECPs as well
        self.orbitalEnergies_ = None #One-electron orbital energies from SCF Calc
        self.canonicalOrbitals_ = None #Canonical orbitals from SCF Calculation
        self.localOrbitals_ = None #Local orbitals from Localization Calculation on Canonical Orbitals
        self.uCanonicalOrbitalsAlpha_ = None #Alpha Canonical Orbitals from unrestricted SCF calculation
        self.uCanonicalOrbitalsBeta_ = None #Beta Canonical Orbitals from unrestricted SCF calculation
        self.naturalOccupations_ = None #Natural orbital occupation numbers
        self.naturalOrbitals_ = None #Natural orbital occupation numbers
        self.fockMatrix_ = None #Fock matrix from SCF calculation

        #Two-electron spatial integrals in atomic orbital basis
        #non-compact, chemists notation (ij|kl)=self.twoElectron_[i,j,k,l]
        self.twoElectron_ = None

        #Two Electron Integrals in molecular orbital basis, possibly local or canonical
        self.moTwoElectron_ = None

        #Reduced Density Matrices and Properties from CISD -- A-Alpha B-Beta Electrons

        self.CISD_OPDM_A_ = None 
        self.CISD_OPDM_B_ = None
        self.CISD_TPDM_AA_ = None
        self.CISD_TPDM_AB_ = None
        self.CISD_TPDM_BB_ = None

        """Amplitudes from CC calculations if desired, these will be stored as in PSI4
        with O(ccupied) and V(irtual) with different index sets, and t1 given as VO and t2
        as VVOO.  All spatial orbital indices.  The orbitals are indexed starting from
		0 as the first occupied orbital and 0 as the first virtual orbital"""

        self.T1IA_ = None
        self.T1ia_ = None
        self.T2IJAB_ = None
        self.T2ijab_ = None
        self.T2IjAb_ = None

        #Molecule Geometry Used in XYZ String
        self.xyz_ = None

    def getM(self):
        """Return number of basis functions"""
        return self.M_

    def getNElectrons(self):
        """Get the number of electrons in the molecule"""
        return self.nElectrons_

    def getNucRep(self):
        """Get the nuclear repulsion energy"""
        return self.nuclearRepulsion_

    def OEITransform(self, M, X):
        """Transform the one-electron matrix M by X"""
        return np.dot(X.T, np.dot(M, X))

    def getOverlap(self, X=None):
        """Get the overlap matrix transformed by basis transformation X"""
        if (X is None):
            return self.overlap_
        else:
            return self.OEITransform(self.overlap_, X)

    def getKinetic(self, X=None):
        """Get the kinetic matrix transformed by basis transformation X"""
        if (X is None):
            return self.kinetic_
        else:
            return self.OEITransform(self.kinetic_, X)

    def getNuclear(self, X=None):
        """Get the nuclear attraction matrix transformed by basis transformation X"""
        if (X is None):
            return self.nuclear_
        else:
            return self.OEITransform(self.nuclear_, X)
    
    def getPseduo(self, X=None):
        """Get the pseduo potential matrix transformed by basis transformation X"""
        if (X is None):
            return self.pseudo_
        else:
            return self.OEITransform(self.pseudo_, X)

    def getTotalCore(self, X=None):
        """Get the total core matrix transformed by basis transformation X"""
        if (X is None):
            return self.totalCore_
        else:
            return self.OEITransform(self.totalCore_, X)

    def getTotalCoreECP(self, X=None):
        """Get the total core matrix including ECPs transformed by basis transformation X"""
        if (X is None):
            return self.totalCoreECP_
        else:
            return self.OEITransform(self.totalCoreECP_, X)

    def getFockMatrix(self, X=None):
        """Get the Fock Matrix transformed by basis transformation X"""
        if (X is None):
            return self.fockMatrix_
        else:
            return self.OEITransform(self.fockMatrix_, X)

    def getOrbitalEnergies(self):
        """Get the Hartree Fock (or Kohn-Sham) orbital energies"""
        return self.orbitalEnergies_

    def getCanonicalOrbitals(self):
        """Get the Hartree Fock (or Kohn-Sham) orbitals in the AO basis"""
        return self.canonicalOrbitals_

    def getNaturalOrbitals(self):
        """Get the Natural Orbitals in the AO Basis, usually produced by FCI in these applications"""
        return self.naturalOrbitals_

    def getTEI(self, X=None):
        """Get the two electron integrals transformed by basis transformation X"""
        if (X is None):
            return self.twoElectron_
        else:
            #N^5 Transformation routine as translated from Daniel Crawfords Website
            #Compound Index Useful for Symmetry Unique Access to TEI
            M = self.M_
            cmpdIndex2 = lambda i, j: i*(i+1)/2 + j if i > j else j*(j+1)/2 + i
            cmpdIndex4 = lambda i, j, k, l: cmpdIndex2(cmpdIndex2(i,j),cmpdIndex2(k,l))

            maxCmpdIndex = cmpdIndex4(M-1, M-1, M-1, M-1) + 1
            TEITransformed = np.zeros((M,M,M,M))
            
            #Load into temp arrays
            Vee = np.zeros(maxCmpdIndex)
            for i in range(M):
                for j in range(M):
                    for k in range(M):
                        for l in range(M):
                            Vee[cmpdIndex4(i,j,k,l)] = self.twoElectron_[i,j,k,l]

            VeeOrthogonal = np.zeros(maxCmpdIndex)
            A = np.zeros((M,M))
            B = np.zeros((M,M))
            TMP = np.zeros( (M*(M+1)/2, M*(M+1)/2) )
            ij = 0
            for i in range(M):
                for j in range(i+1):
                    kl = 0
                    for k in range(M):
                        for l in range(k+1):
                            ijkl = cmpdIndex2(ij,kl)
                            A[k][l] = A[l][k] = Vee[ijkl]
                            kl+=1
                    B = np.dot(X.T, A)
                    A = np.dot(B, X)

                    kl = 0
                    for k in range(M):
                        for l in range(k+1):
                            TMP[kl][ij] = A[k][l]
                            kl += 1
                    ij += 1

            kl = 0
            for k in range(M):
                for l in range(k+1):
                    A = np.zeros((M,M))
                    B = np.zeros((M,M))
                    ij = 0
                    for i in range(M):
                        for j in range(i+1):
                            A[i][j] = A[j][i] = TMP[kl][ij]
                            ij += 1
                    B = np.dot(X.T, A)
                    A = np.dot(B, X)
                    ij = 0
                    for i in range(M):
                        for j in range(i+1):
                            klij = cmpdIndex2(kl,ij)
                            VeeOrthogonal[klij] = A[i][j]
                            ij += 1
                    kl += 1
            #Unload transformed integrals
            for i in range(M):
                for j in range(M):
                    for k in range(M):
                        for l in range(M):
                            TEITransformed[i,j,k,l] = VeeOrthogonal[cmpdIndex4(i,j,k,l)]
            return TEITransformed
