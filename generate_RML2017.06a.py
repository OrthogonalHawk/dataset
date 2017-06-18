#!/usr/bin/env python
import argparse
import logging
from transmitters import transmitters
from source_alphabet import source_alphabet
import analyze_stats
from gnuradio import channels, gr, blocks
import numpy as np
import numpy.fft, cPickle, gzip
import random

DESCRIPTION = \
    "Generate Radio Machine Learning (RML) dataset with a dynamic channel model\n" + \
    " across a range of SNRs. Creates a pickled numpy dataset that contains the\n" + \
    " frequency domain representation of the captured signal.\n" + \
    "\n" + \
    "Output format looks like this:\n" + \
    " {('mod_type', SNR): np.array(num_vectors, 2, FFT_SIZE / 4)}\n"

# assume a symbol rate of 0.96 Msymbols per second; the bit rate will vary based
#  on the modulation type. based on the symbol rate, the transmitters.py file is
#  configured for 4 samples per symbol, which works out to 3.84 Msamples per
#  second.
SAMPLE_RATE_IN_SAMPLES_PER_SECOND = 3.84e6
SAMPLES_PER_SYMBOL = 4

# given the sample rate, a 4192 point FFT provides 937.5 Hz/bin resolution, which
#  is typically within a demodulator's correction window. each FFT is also ~1ms in
#  duration (4096 / 3.84e6) = 0.0010667
FFT_SIZE = 4096
SAVE_DATA_SIZE = FFT_SIZE / SAMPLES_PER_SYMBOL

# define the default SNR range values
DEFAULT_MINIMUM_SNR_VALUE = -6
DEFAULT_MAXIMUM_SNR_VALUE = 20
SNR_VALUE_STEP_SIZE = 2

# define the default number of vectors to generate at each SNR level. CIFAR-10
#  has 6000 samples/class. CIFAR-100 has 600. Somewhere in there seems like
#  right order of magnitude
DEFAULT_NUM_VECTORS = 128

apply_channel = True

dataset = {}

def generate_dataset(args):

    # initialize the pseudorandom number generator
    random.seed()

    for snr in range(args.minimum_snr, args.maximum_snr + 1, SNR_VALUE_STEP_SIZE):

        logging.info("Generating vectors for SNR: %f" % (snr))

        for alphabet_type in transmitters.keys():

            logging.debug("Generator vectors for modulation: %s" % (alphabet_type))

            for i, mod_type in enumerate(transmitters[alphabet_type]):

                # create a new placeholder for the next set of data
                dataset[(mod_type.modname, snr)] = np.zeros([args.num_vectors, 2, SAVE_DATA_SIZE], dtype=np.float32)

                # moar vectors!
                insufficient_modsnr_vectors = True
                modvec_indx = 0
                while insufficient_modsnr_vectors:
                    tx_len = int(10e4)
                    if mod_type.modname == "QAM16":
                        tx_len = int(20e4)
                    if mod_type.modname == "QAM64":
                        tx_len = int(30e4)
                    src = source_alphabet(alphabet_type, tx_len, True)
                    mod = mod_type()
                    fD = 1
                    delays = [0.0, 0.9, 1.7]
                    mags = [1, 0.8, 0.3]
                    ntaps = 8
                    noise_amp = 10**(-snr/10.0)
                    chan = channels.dynamic_channel_model( SAMPLE_RATE_IN_SAMPLES_PER_SECOND,
                                                           0.01, 50, .01, 0.5e3, 8, fD, True,
                                                           4, delays, mags, ntaps, noise_amp, random.random() * random.randint(1, 0x1337))

                    snk = blocks.vector_sink_c()

                    tb = gr.top_block()

                    # connect blocks
                    if apply_channel:
                        tb.connect(src, mod, chan, snk)
                    else:
                        tb.connect(src, mod, snk)
                    tb.run()

                    # retrieve the output vector after it goes through the
                    #  dynamic channel
                    raw_output_vector = np.array(snk.data(), dtype=np.complex64)

                    # start the sampler some random time after channel model transients (arbitrary values here)
                    sampler_indx = random.randint(500, 750)

                    while sampler_indx + FFT_SIZE < len(raw_output_vector) and modvec_indx < args.num_vectors:

                        sampled_vector = raw_output_vector[sampler_indx:sampler_indx + FFT_SIZE]

                        # Normalize the energy in this vector to be 1
                        energy = np.sum((np.abs(sampled_vector)))
                        sampled_vector = sampled_vector / energy

                        # get the frequency domain representation of the baseband signal
                        sampled_vector = np.fft.fft(sampled_vector)

                        # since we have a baseband IQ signal, we can use all of the
                        #  FFT results. circularly shift the data so that the DC
                        #  bin is at index FFT_SIZE / 2
                        sampled_vector = np.roll(sampled_vector, FFT_SIZE / 2)

                        # save the real and imaginary components that correspond to the
                        #  signal of interest. these are the 'center' SAVE_DATA_SIZE
                        #  FFT bins
                        DC_BIN_IDX = FFT_SIZE / 2
                        dataset[(mod_type.modname, snr)][modvec_indx,0,:] = np.real(sampled_vector[DC_BIN_IDX - SAVE_DATA_SIZE/2:DC_BIN_IDX + SAVE_DATA_SIZE/2])
                        dataset[(mod_type.modname, snr)][modvec_indx,1,:] = np.imag(sampled_vector[DC_BIN_IDX - SAVE_DATA_SIZE/2:DC_BIN_IDX + SAVE_DATA_SIZE/2])

                        # bound the upper end very high so it's likely we get multiple passes through
                        # independent channels
                        sampler_indx += random.randint(FFT_SIZE, round(len(raw_output_vector)*.05))

                        # move on to the next vector index
                        modvec_indx += 1

                    if modvec_indx == args.num_vectors:
                        # we're all done
                        insufficient_modsnr_vectors = False

    logging.info("Finished creating the dataset. Writing to disk...")
    cPickle.dump( dataset, file("RML2017.06a_dict.dat", "wb" ) )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--maximum_snr', default=DEFAULT_MAXIMUM_SNR_VALUE, type=int, \
                        help="Maximum SNR for dataset vectors; defaults to %u" % (DEFAULT_MAXIMUM_SNR_VALUE))

    parser.add_argument('--minimum_snr', default=DEFAULT_MINIMUM_SNR_VALUE, type=int, \
                        help="Minimum SNR for dataset vectors; defaults to %u" % (DEFAULT_MINIMUM_SNR_VALUE))

    parser.add_argument('--num_vectors', default=DEFAULT_NUM_VECTORS, type=int, \
                        help="Number of vectors to generate at each SNR; defaults to %u" % DEFAULT_NUM_VECTORS)

    parser.add_argument('--verbose', action='store_true', help='Increase logging verbosity')

    # parse the command-line arguments
    args = parser.parse_args()

    # setup the logger
    active_log_level = logging.INFO
    if args.verbose:
        active_log_level = logging.DEBUG
    logging.basicConfig(level=active_log_level, \
                        format='%(asctime)s [RML_2017.06a] [%(levelname)s] %(message)s', \
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    generate_dataset(args)
