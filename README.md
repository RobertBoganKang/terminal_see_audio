# Terminal See Audio

## Introduction

Similar to `Adobe Audition`. This project will check the audio information quickly inside of terminal. 

The following functions have been implemented:

* split audio;
* plot wave & spectral;
* play audio;
* spiral spectral analyzer;

## Requirement

Python library `matplotlib`, `numpy`, `scipy`, `librosa`, `soundfile`, `pypng`.

Terminal library: `ffmpeg`, `sox`.

`T-IMG` library is needed at [timg](https://github.com/hzeller/timg/) and install.

## Usage tutorial

Simple script will learn.

### Starting

```bash
python see.py <input_file>
```

* `input_file` is the path of input audio file as long as program will recognize;

### Script

* `two numbers`: starting time & ending time;
  * example: `10 20` means to split audio from **10s** to **20s**;
* empty string `''` (press return without input): plot last generated graphics;
* `p`: play audio;
* `q`: quit program;
* `r`: reset starting and ending time;
* `m`: change the `mode` of showing graphics; there are several modes can be changed;
  * `fft` and `fbank` changes the `y` axis transform;
  * `power` and `log` change the color density (spectral values) transform of graphics;
  * `mono` and `stereo` change the channel number to be `1` or `not 1` if have;
  * example: `m fft` means switch to `fft` mode;
* `sr`: change the sample rate of showing audio;
  * example: `sr 16000` means switch to `16000Hz` sample rate;
* `o`: change the input audio file path;
  * example: `o demo/june.ogg` or `o "demo/june.ogg"` means change input audio file to `demo/june.ogg`;
* `tmp`: to change the temp folder for temp files;
* `h`: to print `README.md` file (this file) as `help`;
* `spiral`: spiral analyzer for short period signals by giving a starting time;
  * example: `spiral 10` is to analyze `10~s` spectral, `spiral p` is to play last analyzed result;

### Advanced Script

* `=` (WARNING: may cause fatal crash): to `eval` in python (simple calculation) or to `exec` in python (set system parameters);
  * example: `=1+1` to get answer `2`;
  
  * example: `=self.n_window=512` to set the `n_window` to `512`;
* `sh`: to execute `shell` script;
  * example: `sh echo $PATH` to show `$PATH` variable;

Path can be auto-complete and the space `' '` can be replaced by `\s`.

## Demo

Functions can be tested with `./demo/june.ogg` of piano song (*Tchaikovsky - June Barcarolle Op. 37 No. 6*; I played) with:

```bash
python see.py
```

More clear image can be achieved by setting smaller font size.