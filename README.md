# Audio Cycle GAN

This repository features a Cycle Generative Adversarial Network able to generate
good quality synthesized audio by improving poorly synthesized audio samples.

## Audio samples

TODO

## Setup

This repository requires python3, numpy 1.17.2, torch 1.0.1.post2 and librosa 0.7.0.

By default the training set should be set in the following directory structure:

- data
  - fr
    - test
      - A
      - B
    - train
      - A
      - B

I recommand using the [french CSS dataset](http://kaggle.com/bryanpark/french-single-speaker-speech-dataset) as B. Use any frnech synthetised audio dataset as A (I will provide one later).

## Usage

You can see the help page and run the model on the fr dataset like this:

```python
python3 audiocyclegan.py --help
python3 audiocyclegan.py # trains and generate audio samples
```

It will run the training for 200 epochs and create samples every 10 epochs
