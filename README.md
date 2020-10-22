# Terminal See Audio

## Introduction

Similar to `Adobe Audition`. This project will check the audio information quickly inside of terminal.

This project implemented following functions:

* plot wave & spectral;
* split audio;
* play audio;

## Requirement

Python library `matplotlib`, `numpy`, `scipy`, `librosa`, `soundfile`.

Terminal library: `ffmpeg`, `sox`.

`T-IMG` library is needed at [timg](https://github.com/hzeller/timg/) and install.

## Usage tutorial

Simple script will learn.

### Starting

```bash
python see.py -i <input_file> -sr <sample_rate> -tmp <temp_folder> -m <mode>
```

* `--input`, `-i` input of audio file as long as program will recognize;
* `--sample_rate`, `-sr` sample rate to plot audio;
* `--temp_folder`, `-tmp` (optional) temp folder for audios and graphics to show at the terminal (will delete after use);
* `--mode`, `-m` mode for spectral to plot, `fbank` or `fft`;

### Script

* two numbers: starting time & ending time (example: `0 10` means from **0s** to **10s**);
* empty string `''`: plot last graphics;
* `p`: play audio;
* `q`: quit program;