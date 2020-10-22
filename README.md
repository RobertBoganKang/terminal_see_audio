# Terminal See Audio

## Introduction

Similar to `Adobe Audition`. This project will check the audio information quickly inside of terminal. 

The following functions have been implemented:

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
* `--spectral_transform_y`, `-my` mode for transforming spectral y-location, `fbank` or `fft`;
* `--spectral_transform_v`, `-mv` mode for transforming spectral values, `power` or `log`;

### Script

* two numbers: starting time & ending time (example: `0 10` means from **0s** to **10s**);
* empty string `''`: plot last graphics;
* `p`: play audio;
* `q`: quit program;
* `r`: reset starting and ending time;

### Demo

With `./demo/june.ogg` of piano song (I played), we can test the functions.