# ltl-zero-shot
This is the code repository for the paper "[Encoding formulas as deep networks: Reinforcement learning for zero-shot execution of LTL formulas](https://arxiv.org/abs/2006.01110)"

## Requirements
- Python 3.7
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Spot 2.9.3](https://spot.lrde.epita.fr/install.html)

## Install
We recommend to run in a new virtual environment. Please installed the library listed above before proceeding. You can clone this repo and run `pip install -r requirements.txt` to get the dependencies.

## Run

`run_char.sh` and `run_craft.sh` are the training scripts for the experiments. This will run for a long time as it also runs the test set every 100 formula updates. 
