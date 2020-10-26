# Terminal See Audio

## Introduction

Similar to `Adobe Audition`. This project will check the audio information quickly inside of terminal. 

The following functions have been implemented:

* split audio;
* plot wave & spectral;
* play audio;

## Requirement

Python library `matplotlib`, `numpy`, `scipy`, `librosa`, `soundfile`, `pypng`.

Terminal library: `ffmpeg`, `sox`.

`T-IMG` library is needed at [timg](https://github.com/hzeller/timg/) and install.

## Usage tutorial

Simple script will learn.

### Starting

```bash
python see.py -i <input_file> -tmp <temp_folder>
```

* `--input`, `-i` input of audio file as long as program will recognize;
* `--temp_folder`, `-tmp` (optional) temp folder for audios and graphics to show at the terminal (will delete after use);

### Script

* `two numbers`: starting time & ending time;
  * example: `10 20` means from **10s** to **20s**;
* empty string `''` (press return without input): plot last generated graphics;
* `p`: play audio;
* `q`: quit program;
* `r`: reset starting and ending time;
* `m`: change the `mode` of showing spectral graphics; there are 4 modes can be changed;
  * `fft` and `fbank` changes the `y` axis transform;
  * `power` and `log` change the color density (spectral values) transform of graphics;
  * example: `m fft` means switch to `fft` mode;
* `sr`: change the sample rate of showing audio;
  * example: `sr 16000` means switch to `16000Hz` sample rate;
* `o`: change the input audio file path;
  * example: `o demo/june.ogg` or `o "demo/june.ogg"` means change input audio file to `demo/june.ogg`;

### Advanced Script

* `=` (WARNING: may cause fatal crash): to `eval` in python (simple calculation) or to `exec` in python (set system parameters);
  
  * example: `=1+1` to get answer `2`;
  
  * example: `=self.n_window=512` to set the `n_window` to `512`;
* `sh`: to execute `shell` script;
  
  * example: `sh echo $PATH` to show `$PATH` variable;

## Demo

Functions can be tested with `./demo/june.ogg` of piano song (*Tchaikovsky - June Barcarolle Op. 37 No. 6*; I played) with:

```bash
python see.py
```

More clear image can be achieved by setting smaller font size.