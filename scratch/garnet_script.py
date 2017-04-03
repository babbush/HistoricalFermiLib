import numpy as np
import numpy.fft
import numpy.linalg

def get_Gv(M, L):
    """
    return 2M+1 G vectors for cell of length L
    """
    reciprocal_vector = 2*np.pi/L
    gxrange = np.append(range(M+1), range(-M,0))

    return gxrange * reciprocal_vector

def KE_rspace(M, L):
    """
    build KE in real space
    """
    Gv2 = (get_Gv(M, L)**2.) / 2.

    # We can do a single FFT
    # since FFT'ing the bra and ket is equivalent up to
    # a normalization factor of doing a single FFT
    KE_row = np.fft.ifft(Gv2)
    KE_mat = np.zeros([2*M+1,2*M+1])

    # create matrix by cyclically shifting KE_row (only works in 1D)
    for i in range(2*M+1):
       KE_mat[i,:]=np.roll(KE_row, i)

    return KE_mat

def test():
    np.set_printoptions(precision=4)
    M, L= 2, 2. # M is number of G values, L is length of box
    KE_mat = KE_rspace(M, L)
    print "real space KE\n", KE_mat
    eigvals = np.linalg.eigvalsh(KE_mat)
    Gv2 = (get_Gv(M, L)**2) / 2.
    print 'kinetic diagonals are {}'.format(Gv2)

    seigvals = np.sort(eigvals) # the real space eigenvalues
    sGv2 = np.sort(Gv2) # the k2 values in momentum space
    print 'position eigs are {}.'.format(seigvals)

    print np.linalg.norm(seigvals - sGv2) # should be zero if they are the same


test()

