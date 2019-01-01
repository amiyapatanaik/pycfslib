#!/usr/bin/python
# Electrode fall off detection added
# Updated to support the new NEO prediction system
# Z3Score will no longer support V1 CFS
# New version now includes EMG channel, CFS version is now 2
# additional vectorization applied for higher speeds
# modified to be compatible with python 3
# function to read, write and create CFS stream (byte array) from raw PSG data
# (c)-2018 Neurobit Technologies Pte Ltd - Amiya Patanaik amiya@neurobit.io
#
# Licensed under GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import base64
import hashlib
import struct
import zlib
import warnings
import numpy as np
from io import BytesIO
from skimage.measure import block_reduce
from scipy.signal import firwin, lfilter, resample_poly, stft
from numba import jit


@jit
def create_stream_v2(C3, C4, EOGL, EOGR, EMG, sampling_rates, compressionbit=True, hashbit=True, check_quality = True):
    SRATE = 100  # Hz
    LOWPASS = 35.0  # Hz
    HIGHPASS = 0.3  # Hz
    LOWPASSEOG = 35.0  # Hz
    LOWPASSEMG = 80.0  # Hz
    channels = 5  # 2EEG 2EOG 1EMG

    if (sampling_rates[0] < 100 or sampling_rates[1] < 100 or sampling_rates[2] < 200):
        raise RuntimeError("Sampling rate too low.")

    Fs_EEG = sampling_rates[0] / 2.0
    Fs_EOG = sampling_rates[1] / 2.0
    Fs_EMG = sampling_rates[2] / 2.0

    one = np.array(1)

    bEEG = firwin(51, [HIGHPASS / Fs_EEG, LOWPASS / Fs_EEG], pass_zero=False, window='hamming', scale=True)
    bEOG = firwin(51, [HIGHPASS / Fs_EOG, LOWPASSEOG / Fs_EOG], pass_zero=False, window='hamming', scale=True)
    bEMG = firwin(51, [HIGHPASS / Fs_EMG, LOWPASSEMG / Fs_EMG], pass_zero=False, window='hamming', scale=True)

    eogL = lfilter(bEOG, one, EOGL)
    eogR = lfilter(bEOG, one, EOGR)
    eeg = (lfilter(bEEG, one, C3) + lfilter(bEEG, one, C4)) / 2.0
    emg = lfilter(bEMG, one, EMG)

    if sampling_rates[0] != 100:
        P = 100
        Q = sampling_rates[0]
        eeg = resample_poly(eeg, P, Q)

    if sampling_rates[1] != 100:
        P = 100
        Q = sampling_rates[1]
        eogL = resample_poly(eogL, P, Q)
        eogR = resample_poly(eogR, P, Q)

    if sampling_rates[2] != 200:
        P = 200
        Q = sampling_rates[2]
        emg = resample_poly(emg, P, Q)

    totalEpochs = int(len(eogL) / 30.0 / SRATE)
    data_length = 32 * 32 * (channels - 1) * totalEpochs
    data = np.empty([data_length], dtype=np.float32)
    window_eog = np.hamming(128)
    window_eeg = np.hamming(128)
    window_emg = np.hamming(256)
    epochSize = 32 * 32 * (channels - 1)
    data_frame = np.empty([32, 32, channels - 1])
    mean_power = np.empty([channels - 1, totalEpochs])

    # spectrogram computation
    for i in range(totalEpochs):
        frame1 = stft(eeg[i * 3000:(i + 1) * 3000], window=window_eeg, noverlap=36, boundary=None, nperseg=128,
                      return_onesided=True, padded=False)
        frame2 = stft(eogL[i * 3000:(i + 1) * 3000], window=window_eog, noverlap=36, boundary=None, nperseg=128,
                      return_onesided=True, padded=False)
        frame3 = stft(eogR[i * 3000:(i + 1) * 3000], window=window_eog, noverlap=36, boundary=None, nperseg=128,
                      return_onesided=True, padded=False)
        frame4 = stft(emg[i * 6000:(i + 1) * 6000], window=window_emg, noverlap=71, boundary=None, nperseg=256,
                      return_onesided=True, padded=False)

        data_frame[:, :, 0] = abs(frame1[2][1:33, 0:32]) * np.sum(window_eeg)  # EEG
        data_frame[:, :, 1] = abs(frame2[2][1:33, 0:32]) * np.sum(window_eog)  # EOG-L
        data_frame[:, :, 2] = abs(frame3[2][1:33, 0:32]) * np.sum(window_eog)  # EOG-R
        data_frame[:, :, 3] = block_reduce(abs(frame4[2][1:129, :]) * np.sum(window_emg), (4, 1), np.mean)   # EMG
        mean_power[:, i] = np.mean(data_frame, (0, 1))

        data[i * epochSize:(i + 1) * epochSize] = np.reshape(data_frame, epochSize, order='F')

    
    quality = np.sum(mean_power > 800,1)*100/totalEpochs

    if np.any(quality > 10) and check_quality:
        print("Warning: Electrode Falloff detected, use qc_cfs function to check which channel is problematic")
        
    signature = bytearray(
        struct.pack('<3sBBBBh??', b'CFS', 2, 32, 32, (channels - 1), totalEpochs, compressionbit, hashbit))

    data = data.tostring()

    raw_digest = []
    if hashbit:
        shaHash = hashlib.sha1()
        shaHash.update(data)
        raw_digest = shaHash.digest()

    if compressionbit:
        data = zlib.compress(data)

    if hashbit:
        stream = signature + raw_digest + data
    else:
        stream = signature + data

    return stream


def save_stream_v2(file_name, C3, C4, EOGL, EOGR, EMG, sampling_rates, compressionbit=True, hashbit=True, check_quality=True):
    stream = create_stream_v2(C3, C4, EOGL, EOGR, EMG, sampling_rates, compressionbit=compressionbit, hashbit=hashbit, check_quality=check_quality)

    with open(file_name, 'wb') as f:
        f.write(stream)

    return stream


@jit
def create_stream(EEG_data, sampling_rate, compressionbit=True, hashbit=True, check_quality=True):
    warnings.warn('You are using version 1 of CFS, CFS version 1 is deprecated and will not be supported by Z3Score in the future.', RuntimeWarning)
    SRATE = 100 #Hz
    LOWPASS = 45.0 #Hz
    HIGHPASS = 0.3 #Hz
    LOWPASSEOG = 12.0 #Hz
    Fs = sampling_rate/2.0
    one = np.array(1)
    bEEG = firwin(51, [HIGHPASS / Fs, LOWPASS / Fs], pass_zero=False, window='hamming', scale=True)
    bEOG = firwin(51, [HIGHPASS / Fs, LOWPASSEOG / Fs], pass_zero=False, window='hamming', scale=True)
    eogL = lfilter(bEOG,one,EEG_data[2,:])
    eogR = lfilter(bEOG,one,EEG_data[3,:])
    eeg = (lfilter(bEEG,one,EEG_data[0,:]) + lfilter(bEEG,one,EEG_data[1,:]))/2.0

    if sampling_rate != 100:
        P = 100
        Q = sampling_rate
        eogL = resample_poly(eogL,P,Q)
        eogR = resample_poly(eogR,P,Q)
        eeg = resample_poly(eeg,P,Q)

    totalEpochs = int(len(eogL)/30.0/SRATE)
    data_length = 32*32*3*totalEpochs
    mean_power = np.empty((3, totalEpochs))
    data = np.empty([data_length], dtype=np.float32)
    window = np.hamming(128)
    epochSize = 32 * 32 * 3

    #STFT based spectrogram computation
    for i in range(totalEpochs):
        for j in range(0,3000-128-1,90):
            tIDX = int(j/90)
            frame1 = abs(np.fft.fft(eeg[i * 3000 + j: i * 3000 + j + 128]*window))
            frame2 = abs(np.fft.fft(eogL[i * 3000 + j: i * 3000 + j + 128] * window))
            frame3 = abs(np.fft.fft(eogR[i * 3000 + j: i * 3000 + j + 128] * window))
            mean_power[:, i] = [np.mean(frame1), np.mean(frame2), np.mean(frame3)]
            data[i*epochSize + tIDX * 32: i*epochSize + tIDX * 32 + 32] = frame1[0:32]
            data[i*epochSize + 32 * 32 + tIDX * 32: i*epochSize + 32 * 32 + tIDX * 32 + 32] = frame2[0:32]
            data[i*epochSize + 32 * 32 * 2 + tIDX * 32: i*epochSize + 32 * 32 * 2 + tIDX * 32 + 32] = frame3[0:32]


    quality = np.sum(mean_power > 800,1)*100/totalEpochs

    if np.any(quality > 10) and check_quality:
        print("Warning: Electrode Falloff detected, use qc_cfs function to check which channel is problematic")

    signature = bytearray(struct.pack('<3sBBBBh??', b'CFS',1,32,32,3,totalEpochs,compressionbit,hashbit))
    data = data.tostring()

    raw_digest = []
    if hashbit:
        shaHash = hashlib.sha1()
        shaHash.update(data)
        raw_digest = shaHash.digest()
    
    if compressionbit:
        data = zlib.compress(data)

    if hashbit:
        stream = signature + raw_digest + data
    else:
        stream = signature + data

    return stream


def save_stream(file_name, EEG_data, sampling_rate, compressionbit=True, hashbit=True, check_quality=True):

    stream = create_stream(EEG_data, sampling_rate, compressionbit, hashbit, check_quality)

    with open(file_name, 'wb') as f:
        f.write(stream)

    return stream


def read_stream(stream, check_quality = True):
    # Read header:
    # 3 bytes signature, 1 byte version, 1 byte frequency, 1 byte time, 1 byte channel 2 bytes epochs
    # 1 byte compressionbit, 1 byte hashbit
    bytes = stream.read(11)
    header = struct.unpack('<3sBBBBh??', bytes)
    urlSafeHash = None
    digest = None

    if (header[0].decode("ascii") != 'CFS'):
        raise RuntimeError("File is not a valid CFS file.")

    if (header[1] != 1 and header[1] != 2):
        raise RuntimeError("Invalid CFS version.")

    version = header[1]
    nfreq = header[2]
    ntime = header[3]
    nchannel = header[4]
    nepoch = header[5]
    compressionbit = header[6]
    hashbit = header[7]

    if (hashbit):
        # read SHA hash
        bytes = stream.read(20)
        digest = struct.unpack('<20B', bytes)
        urlSafeHash = base64.urlsafe_b64encode(("".join(map(chr, digest))).encode('UTF-8')).decode('ascii')
        urlSafeHash = urlSafeHash[0:-1]

    # read rest of the data
    bytes = stream.read()

    if (compressionbit):
        bytes = zlib.decompress(bytes)

    rawStream = struct.unpack('<' + str(nfreq * ntime * nchannel * nepoch) + 'f', bytes)

    if (hashbit):
        shaHash = hashlib.sha1()
        shaHash.update(bytes)
        rawDigest = shaHash.digest()
        msgDigest = [c for c in rawDigest]

        if (tuple(msgDigest) != digest):
            raise RuntimeError("File is corrupt.")

    dataStream = np.asarray(rawStream, dtype='float32')
    dataStream = np.reshape(dataStream, (nfreq, ntime, nchannel, nepoch), order="F")

    quality = np.sum(np.mean(dataStream,(0,1)) > 800,1)*100/np.shape(dataStream)[-1]
    if np.any(quality > 10) and check_quality:
        print("Warning: Electrode Falloff detected, use qc_cfs function to check which channel is problematic")


    header = {'freq': nfreq, 'time': ntime, 'channel': nchannel, 'version': version,
              'epoch': nepoch, 'compression': compressionbit, 'hash': hashbit, 'url': urlSafeHash}

    return dataStream, header


def read_header(cfs_file):
    # Read header:
    # 3 bytes signature, 1 byte version, 1 byte frequency, 1 byte time, 1 byte channel 2 bytes epochs
    # 1 byte compressionbit, 1 byte hashbit
    stream = open(cfs_file, "rb")
    bytes = stream.read(11)
    header = struct.unpack('<3sBBBBh??', bytes)

    if (header[0].decode("ascii") != 'CFS'):
        raise RuntimeError("File is not a valid CFS file.")

    if (header[1] != 1 and header[1] != 2):
        raise RuntimeError("Invalid CFS version.")

    version = header[1]
    nfreq = header[2]
    ntime = header[3]
    nchannel = header[4]
    nepoch = header[5]
    compressionbit = header[6]
    hashbit = header[7]

    urlSafeHash = None

    if (hashbit):
        # read SHA hash
        bytes = stream.read(20)
        digest = struct.unpack('<20B', bytes)
        urlSafeHash = base64.urlsafe_b64encode(("".join(map(chr, digest))).encode('UTF-8')).decode('ascii')
        urlSafeHash = urlSafeHash[0:-1]

    stream.seek(0)

    header = {'freq': nfreq, 'time': ntime, 'channel': nchannel, 'version': version,
              'epoch': nepoch, 'compression': compressionbit, 'hash': hashbit, 'url': urlSafeHash}
    return header


def read_cfs(cfs_file, check_quality = True):
    stream = open(cfs_file, "rb")
    return read_stream(stream, check_quality)


def qc_cfs(cfs_file, threshold = 10):
    data, header = read_cfs(cfs_file, check_quality = False)
    quality = np.sum(np.mean(data,(0,1)) > 800,1)*100/np.shape(data)[-1]
    electrodes = {
        0: "C3/C4",
        1: "EOG-left",
        2: "EOG-right",
        3: "EMG"
    }
    status = False
    qc = quality > threshold
    idx = np.flatnonzero(qc)
    failed_channels = " "
    message = "All channels passed quality checks."
    
    if np.any(qc):
        status = True
        for i in idx:
            failed_channels += electrodes[i]
        message = "The following channel(s) failed quality checks:" + failed_channels

    return status, quality, message


def qc_stream(bytestream, threshold = 10):
    file_stream = BytesIO(bytestream)
    data, header = read_stream(file_stream, check_quality = False)
    quality = np.sum(np.mean(data,(0,1)) > 800,1)*100/np.shape(data)[-1]
    electrodes = {
        0: "C3/C4",
        1: "EOG-left",
        2: "EOG-right",
        3: "EMG"
    }
    status = False
    qc = quality > threshold
    idx = np.flatnonzero(qc)
    failed_channels = " "
    message = "All channels passed quality checks."

    if np.any(qc):
        status = True
        for i in idx:
            failed_channels += electrodes[i]
        message = "The following channel(s) failed quality checks:" + failed_channels

    return status, quality, message
