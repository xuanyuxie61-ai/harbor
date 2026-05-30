# -*- coding: utf-8 -*-

import numpy as np




HBAR_C         = 197.3269804
MASS_NUCLEON   = 938.918
MASS_PROTON    = 938.2720813
MASS_NEUTRON   = 939.5654133
ELECTRON_MASS  = 0.5109989461
SPEED_OF_LIGHT = 299792458.0
FINE_STRUCTURE = 1.0 / 137.035999084




FM_TO_M        = 1.0e-15
MEV_TO_JOULE   = 1.602176634e-13
SECOND_TO_FS   = 1.0e15




LDA_VOLUME     = 15.75
LDA_SURFACE    = 17.8
LDA_COULOMB    = 0.711
LDA_ASYMMETRY  = 23.7
LDA_PAIRING    = 11.18




WS_V0          = -51.0
WS_R0          = 1.27
WS_A           = 0.67
WS_VSO         = -0.44 * WS_V0
WS_RSO         = 1.27
WS_ASO         = 0.67




G_PAIRING      = 25.0 / 41.0




DEFAULT_FEKETE_DEGREE = 7
DEFAULT_SPARSE_LEVEL  = 4




def reduced_mass(m1, m2):
    return (m1 * m2) / (m1 + m2)


def hbar2_over_2m(mass=MASS_NUCLEON):
    return (HBAR_C ** 2) / (2.0 * mass)
