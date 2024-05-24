# RL Traffic Controller

## Warning

Right now, this project doesn't work on Microsoft Windows because SUMO (`sumo-gui`) will randomly hang when
capturing a screenshot of the simulation using `traci`.

## Setup

1. Install Python version 3.11.4
2. Install the reqiured Python libraries

   ```bash
   pip3.11 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Usage

```text
$ python3.11 run.py --help

usage: run.py [-h] [--arch {v1,v2,v3,v4,v5,v6,v7}] [-c] [-s] [-e N] [--images  [...]]
              mode

positional arguments:
  mode                  train or eval or demo or dry-run

options:
  -h, --help            show this help message and exit
  --arch {v1,v2,v3,v4,v5,v6,v7}
                        ID of the network architecture to use (default: v7)
  -c, --continue        load the saved network and continue training
  -s, --save            save the network after every training episode
  -e N, --episodes N    number of episodes sampled during training (default: 1)
  --images  [ ...]      paths of images (observations), and/or directories containing
                        images, to test the agent on
```

To train the agent from scratch, run the following command and replace `N` with the number of episodes you want. Remove `--save` if you don't want to save the Q network after every episode.

```bash
python3.11 run.py train --episodes N --save
```

To continue training the agent using a previously saved Q network, run the following command.

```bash
python3.11 run.py train --episodes N --continue
```

To see the agent's action values (using a previously saved Q network), run the following command and provide as many images, and directories containing multiple images, as you want. The directories will be searched for images in the first level only. The `--images` option should be the last option used in the command.

```bash
python3.11 run.py eval --images firstPicture.png secondPicture.png imagesDir/ ...
```

To see the agent's action values after the training is finished, provide the images and/or directories when you start the training.

```bash
python3.11 run.py train --episodes N --images firstPicture.png secondPicture.png imagesDir/ ...
```

To demo the agent in a running environment use the following command.

```bash
python3.11 run.py demo
```

If you want to test the code without using SUMO, use the `dry-run` mode to simulate the training process. It functions exactly as the `train` mode.

```bash
python3.11 run.py dry-run
```

### Notes

- The agent can use a variety of network architectures, available in `rl_traffic_controller/networks.py`. Use `python3.11 run.py --help` to find out how to select a specific architecture.
- The agent saves the Q network and related variables after each episode in the `models/` directory.
- The `logs/run.log` provides extensive logs during the training process.
- During training, it's preferable to keep the simulation window visible, as it will make capturing screenshots noticably faster.
- Network configurations depend on the size and format of the input image, thus some configurations may not work with the current settings (specified in `rl_traffic_controller/consts.py`).
