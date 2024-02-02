# ml-in-bioinformatics
This repository contains implementations of solutions to assignments for the course [CSCI4969-6969 Machine Learning in Bioinformatics](https://www.cs.rpi.edu/~zaki/courses/mlib/) taught by Dr. Mohammed J. Zaki at the Rensselaer Polytechnic Institute.


## Requirements
Assignments 1 and 2 train models on JUND transcription factor binding data. See [download_data_JUND.sh](download_data_JUND.sh) for ean example of how to download this data to a local directory. To run the assignment notebooks set the following environment variables to either S3 URIs or paths to local directories (e.g. `s3://<bucket>/<folder>`, `/<local>/<path>`).
```
DATA_DIR_TRAIN 
DATA_DIR_VALIDATION
DATA_DIR_TEST
```
