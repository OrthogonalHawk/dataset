#!/usr/bin/python
#!/bin/env python

import argparse
import logging
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from scipy import signal

def calc_vec_energy(vec):
    isquared = np.power(vec[0],2.0)
    qsquared = np.power(vec[1], 2.0)
    inst_energy = np.sqrt(isquared+qsquared)
    return sum(inst_energy)

def calc_mod_energies(ds):
    for modulation, snr in ds:
        avg_energy = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_energy += calc_vec_energy(vec)
        avg_energy /= nvectors
        print "%s at %i has %i vectors avg energy of %2.1f" % (modulation, snr, nvectors, avg_energy)

def calc_mod_bias(ds):
    for modulation, snr in ds:
        avg_bias_re = 0
        avg_bias_im = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_bias_re += (np.mean(vec[0]))
            avg_bias_im += (np.mean(vec[1]))
        #avg_bias_re /= nvectors
        #avg_bias_im /= nvectors
        print "%s at %i has %i vectors avg bias of %2.1f + %2.1f j" % (modulation, snr, nvectors, avg_bias_re, avg_bias_im)

def calc_mod_stddev(ds):
    for modulation, snr in ds:
        avg_stddev = 0
        nvectors = ds[(modulation,snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_stddev += np.abs(np.std(vec[0]+1j*vec[1]))
        #avg_stddev /= nvectors
        print "%s at %i has %i vectors avg stddev of %2.1f" % (modulation, snr, nvectors, avg_stddev)

def open_ds(location="RML2017.06a_dict.dat"):
    f = open(location)
    ds = cPickle.load(f)
    return ds

def main():

    parser = argparse.ArgumentParser(description="Analyzes RML dataset statistics")

    parser.add_argument('--verbose', action='store_true', help='Increase logging verbosity')

    # parse the command-line arguments
    args = parser.parse_args()

    # setup the logger
    active_log_level = logging.INFO
    if args.verbose:
        active_log_level = logging.DEBUG
    logging.basicConfig(level=active_log_level, \
                        format='%(asctime)s [RML_STATS] [%(levelname)s] %(message)s', \
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    ds = open_ds()
    plt.plot(abs(ds[('PAM4', 20)][0][0][:]))

    #f, Pxx_den = signal.periodogram(ds[('QPSK', 20)][0][0][:])
    #plt.semilogy(f, Pxx_den)
    #plt.plot(ds[('QPSK', 20)][0][1][:])
    plt.show()

    plt.plot(abs(ds[('GFSK', 20)][0][0][:]))
    plt.show()
    #calc_mod_energies(ds)
    #calc_mod_stddev(ds)
    #calc_mod_bias(ds)

if __name__ == "__main__":
    main()
