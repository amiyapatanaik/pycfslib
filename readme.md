# pycfslib

pycfslib is a python library to read, write and create compressed feature set (CFS) file/stream from raw PSG data. Python 3 is now supported. The CFS format is an open standard for communication with the Z3Score and NEO sleep scoring system (https://z3score.com). Instead of using polysomnography data in European Data Format (EDF, https://en.wikipedia.org/wiki/European_Data_Format), the Z3Score system uses CFS files. CFS files are on an average 18X smaller than corresponding EDF files. This reduces data overhead significantly. The format does not allow any user identifiable information ensuring anonymity. The code is released under GPL v3. 
(c)-2018 Neurobit Technologies Pte Ltd - Amiya Patanaik 

### Updates
Added electrode fall off detection. Two new functions added, qc_cfs and qc_stream

Now supports CFS v2, CFS v2 is a new specification that works with the NEO prediction system. NEO replaces the Z3Score prediction system which is now deprecated.  
### Installation

```sh
    pip install pycfslib
```
### Usage
Z3Score provides a RESTful API to access the sleep scoring services. Read about the API here: https://github.com/neurobittechnologies/z3score-api 
Sample code using the CFS library is provided in the repository. 
### Important Functions
```python
    save_stream_v2(file_name, C3, C4, EOGL, EOGR, EMG, sampling_rates, compressionbit=True, hashbit=True, check_quality=True)
```
  - Returns a CFS Version 2 binary stream
  - file_name: is the file name where you want to store the CFS stream. Should have .cfs extension.
  - C3: C3-A2 EEG data, must be sampled at 100Hz or more
  - C4: C4-A1 EEG data, must be sampled at 100Hz or more
  - EOGL: EoGleft-A2 data, must be sampled at 100Hz or more
  - EOGR: EoGright-A1 data, must be sampled at 100Hz or more
  - EMG: chin EMG data, must be sampled at 200Hz or more
  - sampling_rate: is a list of size three with sampling rates of EEG, EOG and EMG data
  - compressionbit: is True (default) if compression is enabled, False otherwise
  - hashbit: is True (default) if a payload SHA1 signature is included in the CFS stream, False otherwise
  - check_quality=True (default) does a quality check (will show warnings if check fails)
  
```python
    create_stream_v2(C3, C4, EOGL, EOGR, EMG, sampling_rates, compressionbit=True, hashbit=True, check_quality=True)
```
  - Returns a CFS Version 2 binary stream
  - C3: C3-A2 EEG data, must be sampled at 100Hz or more
  - C4: C4-A1 EEG data, must be sampled at 100Hz or more
  - EOGL: EoGleft-A2 data, must be sampled at 100Hz or more
  - EOGR: EoGright-A1 data, must be sampled at 100Hz or more
  - EMG: chin EMG data, must be sampled at 200Hz or more
  - sampling_rate: is a list of size three with sampling rates of EEG, EOG and EMG data
  - compressionbit: is True (default) if compression is enabled, False otherwise
  - hashbit: is True (default) if a payload SHA1 signature is included in the CFS stream, False otherwise
  - check_quality=True (default) does a quality check (will show warnings if check fails)
  
```python
    stream = save_stream(file_name, EEG_data, sampling_rate, compressionbit=True, hashbit=True, check_quality=True)
```
  - **WARNING: Deprecated**
  - Returns a CFS binary stream and saves this stream in file_name
  - file_name: is the file name where you want to store the CFS stream. Should have .cfs extension.
  - EEG_data: is a 4 channels X N sample numpy array. The 4 channel in order are C3-A1, C4-A2, EoGleft-A1 and EoGright-A2. Data must be sampled at 100 Hz or more. 
  - sampling_rate: is the signal sampling rate in Hz. All 4 channels must be sampled at the same rate.
  - compressionbit: is True (default) if compression is enabled, False otherwise
  - hashbit: is True (default) if a payload SHA1 signature is included in the CFS stream, False otherwise
  - check_quality=True (default) does a quality check (will show warnings if check fails)

```python
    stream = create_stream(EEG_data, sampling_rate, compressionbit=True, hashbit=True, check_quality=True)
```
  - **WARNING: Deprecated**
  - Returns a CFS binary stream
  - EEG_data: is a 4 channels X N sample numpy array. The 4 channel in order are C3-A1, C4-A2, EoGleft-A1 and EoGright-A2. Data must be sampled at 100 Hz or more. 
  - sampling_rate: is the signal sampling rate in Hz. All 4 channels must be sampled at the same rate.
  - compressionbit: is True (default) if compression is enabled, False otherwise
  - hashbit: is True (default) if a payload SHA1 signature is included in the CFS stream, False otherwise
  - check_quality=True (default) does a quality check (will show warnings if check fails)

```python
    dataStream, header = read_stream(stream, check_quality=True)
```
  - Returns the data as a 4D numpy array (frequencyXtimeXchannelXepochs) and header
  - stream: is the CFS data byte stream
  - check_quality=True (default) does a quality check (will show warnings if check fails)

```python
    dataStream, header = read_cfs(cfs_file, check_quality=True)
```
  - Returns the data as a 4D numpy array (frequencyXtimeXchannelXepochs) and header
  - cfs_file: full path to CFS file
  - check_quality=True (default) does a quality check (will show warnings if check fails)

```python
    header = read_header(stream):
```
  - Returns header of the CFS file
  - stream: CFS byte stream

```python
    status, quality, message = qc_cfs(cfs_file, threshold = 10):
```
  - Returns status (a bool) which is true if QC fails. quality: a vector of size num_channels with percentage of epochs failing quality checks
  - and message: a user summary 
  - cfs_file: full path to CFS file
  - threshold minimum % of epochs which must be low quality to consider a QC fail

```python
    status, quality, message = qc_stream(stream, threshold = 10):
```
  - Returns status (a bool) which is true if QC fails. quality: a vector of size num_channels with percentage of epochs failing quality checks
  - and message: a user summary 
  - stream: CFS byte stream
  - threshold minimum % of epochs which must be low quality to consider a QC fail

 
License
----

GPL V3
