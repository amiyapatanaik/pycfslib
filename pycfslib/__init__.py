#!/usr/bin/python
# modified to be compatible with python 3
# function to read, write and create CFS stream (byte array) from raw PSG data
# Patents pending (c)-2016 Amiya Patanaik amiyain@gmail.com
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
import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


def create_stream(EEG_data, sampling_rate, compressionbit=True, hashbit=True):
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
            data[i*epochSize + tIDX * 32: i*epochSize + tIDX * 32 + 32] = frame1[0:32]
            data[i*epochSize + 32 * 32 + tIDX * 32: i*epochSize + 32 * 32 + tIDX * 32 + 32] = frame2[0:32]
            data[i*epochSize + 32 * 32 * 2 + tIDX * 32: i*epochSize + 32 * 32 * 2 + tIDX * 32 + 32] = frame3[0:32]


    signature = bytearray(struct.pack('<3sBBBBh??', 'CFS',1,32,32,3,totalEpochs,compressionbit,hashbit))
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


def save_stream(file_name, EEG_data, sampling_rate, compressionbit=True, hashbit=True):

    stream = create_stream(EEG_data, sampling_rate, compressionbit, hashbit)

    with open(file_name, 'wb') as f:
        f.write(stream)

    return stream


def read_stream(stream):
    # Read header:
    # 3 bytes signature, 1 byte version, 1 byte frequency, 1 byte time, 1 byte channel 2 bytes epochs
    # 1 byte compressionbit, 1 byte hashbit
    bytes = stream.read(11)
    header = struct.unpack('<3sBBBBh??', bytes)
    urlSafeHash = None
    digest = None

    if (header[0].decode("ascii")  != 'CFS'):
        raise RuntimeError("File is not a valid CFS file.")

    if (header[1] != 1):
        raise RuntimeError("Invalid CFS version.")

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

    header = {'freq': nfreq, 'time': ntime, 'channel': nchannel,
              'epoch': nepoch, 'compression': compressionbit, 'hash': hashbit, 'url': urlSafeHash}

    return dataStream, header


def read_header(stream):
    # Read header:
    # 3 bytes signature, 1 byte version, 1 byte frequency, 1 byte time, 1 byte channel 2 bytes epochs
    # 1 byte compressionbit, 1 byte hashbit
    bytes = stream.read(11)
    header = struct.unpack('<3sBBBBh??', bytes)

    if (header[0] != 'CFS'):
        raise RuntimeError("File is not a valid CFS file.")

    if (header[1] != 1):
        raise RuntimeError("Invalid CFS version.")

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

    header = {'freq': nfreq, 'time': ntime, 'channel': nchannel,
              'epoch': nepoch, 'compression': compressionbit, 'hash': hashbit, 'url': urlSafeHash}
    return header


def read_cfs(cfs_file):

    stream = open(cfs_file, "rb")
    return read_stream(stream)

