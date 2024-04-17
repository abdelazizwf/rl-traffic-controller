# RL Traffic Controller

## Warning

Right now, this project doesn't work on Microsoft Windows because SUMO (`sumo-gui`) will randomly hang when
capturing a screenshot of the simulation using `traci`.

## Setup

1. Install Python version 3.11.4
2. Install [SUMO](https://eclipse.dev/sumo/) version 1.19.0
3. Install the reqiured Python libraries

   ```bash
   pip3.11 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

4. Create required directories

    ```bash
    mkdir logs models
    ```

## Usage

```text
$ python3.11 run.py --help

usage: run.py [-h] [-c] [-s] [-e N] mode {v1,v2,v3,v4,v5,v6,v7} [image_paths ...]

positional arguments:
  mode                  train or eval or demo
  {v1,v2,v3,v4,v5,v6,v7}
                        ID of the network architecture to use
  image_paths           paths of images (observations), and/or directories containing
                        images, to test the agent on

options:
  -h, --help            show this help message and exit
  -c, --continue        load the saved network and continue training
  -s, --save            save the network after every training episode
  -e N, --episodes N    number of episodes sampled during training (default: 50)
```

The agent can use a variety of network architectures, available in `rl_traffic_controller/networks.py`, one of them must be selected in the command line. The list of available architectures can be viewed in the help message, `python3.11 run.py --help`. The commands below use the `v7` architecture.

To train the agent from scratch, run the following command and replace `N` with the number of episodes you want. Remove `--save` if you don't want to save the Q network after every episode.

```bash
python3.11 run.py train v7 --episodes N --save
```

To continue training the agent using a previously saved Q network, run the following command.

```bash
python3.11 run.py train v7 --episodes N --continue
```

To see the agent's action values (using a previously saved Q network), run the following command and provide as many images, and directories containing multiple images, as you want. The directories will be searched for images in the first level only.

```bash
python3.11 run.py eval v7 firstPicture.png secondPicture.png imagesDir/ ...
```

To see the agent's action values after the training is finished, provide the images and/or directories when you start the training.

```bash
python3.11 run.py train v7 firstPicture.png secondPicture.png imagesDir/ ... --episodes N
```

To demo the agent in a running environment use the following command.

```bash
python3.11 run.py demo v7
```

### Notes

- The agent saves the Q network and related variables after each episode in the `models/` directory.
- The `logs/run.log` provides extensive logs during the training process.
- During training, it's preferable to keep the simulation window visible, as it will make capturing screenshots noticably faster.
- Network configurations depend on the size and format of the input image, thus some configurations may not work with the current settings (specified in `rl_traffic_controller/consts.py`).
