# Terminal See Audio

## Introduction

Similar to `Adobe Audition`, this project will check the audio information quickly inside of terminal. 

## Requirement

Python library can be installed with `pip install -r requrements.txt`.

Terminal library: `ffmpeg`, `sox`.

### Image Viewer Library

There are several options:

* `T-IMG` library can be found at [timg](https://github.com/hzeller/timg/) and install (need to set parameter `self.plot_command='timg {}'`);
* `Terminal Image Viewer (TIV)` library can be found at [tiv](https://github.com/stefanhaustein/TerminalImageViewer) and install (default);

## Usage tutorial

Simple script will learn.

### Starting

```bash
python see.py <input_file>
```

* `input_file` is the path of input audio file as long as program will recognize;

### Script

#### Basic

* `two numbers`: starting time & ending time;
  * example: `10 20` means to split audio from `10s` to `20s`;
* empty string `''` (press return without input): plot last generated graphics;
* `p`: play last analyzed audio;
  * `pp`: play the partition of audio where used for analyzers;
* `q`: quit program;
* `r`: reset starting and ending time;
* `m`: change the `mode` of showing graphics; there are several modes can be changed;
  * `fft` and `fbank` changes the `y` axis transform;
  * `power` and `log` change the color density (spectral values) transform of graphics;
  * `mono` and `stereo` change the channel number to be `1` or `not 1` if have;
  * `spectral` and `entropy`: to plot spectral or Shannon's entropy;
  * `color` and `nocolor`: some analyzer will be colorful by its pitch or not;
  * example: `m fft` means switch to `fft` mode;
* `nonlinear`: simulate the nonlinear effect due to the physical structure of ear/cochlea when hearing a very loud sound;
  * `number`: nonlinear coefficient (sound louder, number will be larger);
  * example: `nonlinear 5` will transform audio into nonlinear form at the parameter of `5`;
* `sr`: change the sample rate of showing audio;
  * example: `sr 16000` means switch to `16000Hz` sample rate;
* `fr`: change the video frame rate for analyzer;
  * example: `fr 30` means switch to `30Hz` video frame rate;
* `o`: change the input audio file path;
  * example: `o demo/june.ogg` or `o "demo/june.ogg"` means change input audio file to `demo/june.ogg`;
* `h`: to print `README.md` file (this file) as `help`;

#### Analyzer

* analyzers for short period signals, or to generate videos;
  * `#`: piano, `#=`: chroma gram piano, `@`: spiral, `*%`: phase, `*<`: source angle, `*-*`: source location, `|`: strings, `&`: lattice;
    * `number`: `@10` is to analyze `10~s` spectral;
    * `''`: empty is to plot last analyzed analyzer graphics;
    * `two numbers`: generate video of analyzer at starting time & ending time;
    * `*`: play last calculated analyzer video result;
* `##`: to plot piano roll, `##=`: chroma gram piano roll;
  * `two numbers`: plot piano roll at starting time & ending time; `#10 20` will calculate the piano roll of `10s` to `20s`;
  * `''`: plot last calculated piano roll result;
* `^`: calculate the spectral tuning peaks frequencies (peaks) and plot;
  * `number`: `^10` is to extract `10~s` peaks frequencies components, music note names, and its frequency power at combined channel `*` and separate channel `number`;
  * `two numbers`: `starting time` + `tuning frequency (Hz)`, then plot tuning graphics, `^0 99.041Hz` or `^0 G2+18.31c` to plot tuning graphics at `0~s` with tuning frequency `99.041Hz` or `G2+18.31c`;
* `>`: to play sine wave the given frequency;
  * `number+Hz`: to play sound directly from number of frequency; `>440Hz` is to play sound at `440Hz`;
  * `music notes name`: to play sound translate from music notes; `>a4` is to play `a4` notes; `>a4-50c` is to play sound at `a4` but with `50` cents lower;
  * `number+m`: define sound source distance;
  * `number+rad` or `number+deg`: define sound source angle;
  * `number+s`: define duration to play;

#### Programming

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

## Language

English|[中文](doc/README_zh.md)