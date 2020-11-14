# Terminal See Audio

## Introduction

Similar to `Adobe Audition`. This project will check the audio information quickly inside of terminal. 

The following functions have been implemented:

* split audio;
* plot wave & spectral;
* play audio;
* spiral/piano spectral analyzer;

## Requirement

Python library `matplotlib`, `numpy`, `scipy`, `librosa`, `soundfile`, `pypng`, `peakutils`.

Terminal library: `ffmpeg`, `sox`.

### Image Viewer Library

There are several options:

* `T-IMG` library can be found at [timg](https://github.com/hzeller/timg/) and install (default);
* `Terminal Image Viewer (TIV)` library can be found at [tiv](https://github.com/stefanhaustein/TerminalImageViewer) and install (need to set parameter `self.plot_command='tiv {}'`);

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
  * `p*`: play the partition of audio where used `@`, `#` or `^` to analyze spectral;
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
* `@` or `#`: spiral/piano analyzer for short period signals by giving a starting time;
  * number: `@10` is to analyze `10~s` spectral;
  * `''`: empty is to plot last analyzed spiral graphics;
  * if command `#` follows two numbers, it will plot piano roll at starting time & ending time; `#10 20` will calculate the piano roll of `10s` to `20s`;
  * command `##` will plot last calculated piano roll result;
* `^`: calculate the spectral tuning peaks frequencies (peaks);
  * number: `^10` is to extract `10~s` peaks frequencies components, music note names, and its frequency power at combined channel `*` and separate channel `number`;

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