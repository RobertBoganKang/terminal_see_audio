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
python see.py -i <input_file> -sr <sample_rate> -tmp <temp_folder>
```

* `--input`, `-i` input of audio file as long as program will recognize;
* `--sample_rate`, `-sr` sample rate to plot audio;
* `--temp_folder`, `-tmp` (optional) temp folder for audios and graphics to show at the terminal (will delete after use);

### Script

* `two numbers`: starting time & ending time (example: `10 20` means from **10s** to **20s**);
* empty string `''` (press return without input): plot last generated graphics;
* `p`: play audio;
* `q`: quit program;
* `r`: reset starting and ending time;
* `mode`: change the mode of showing spectral graphics; there are 4 modes can be changed;
  * `fft` and `fbank` changes the `y` axis transform;
  * `power` and `log` change the color density transform of graphics;

### Demo

Functions can be tested with `./demo/june.ogg` of piano song (*Tchaikovsky - June Barcarolle Op. 37 No. 6*; I played).

If image is blurry, please set smaller font size to get more clear image.