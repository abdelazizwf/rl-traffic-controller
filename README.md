# RL Traffic Controller

## Warning

Right now, this project doesn't work on Microsoft Windows because SUMO (`sumo-gui`) will randomly hang when
capturing a screenshot of the simulation using `traci`.

## Setup

1. Install Python version 3.11.4
2. Install [SUMO](https://eclipse.dev/sumo/) version 1.19.0
3. Install the reqiured Python libraries

   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

4. Create required directories

    ```bash
    mkdir logs data models
    ```

## Usage

```text
$ python run.py -h

usage: run.py [-h] [-c] [-r] [-e N] mode {v1,v2,v3,v4,v5} [image_paths ...]

positional arguments:
  mode                train or eval or demo
  {v1,v2,v3,v4,v5}    ID of the network architecture to use
  image_paths         paths of images (observations), and/or directories containing
                      images, to test the agent on

options:
  -h, --help          show this help message and exit
  -c, --continue      load the saved network and continue training
  -r, --remote        setup the VNC client connection
  -e N, --episodes N  number of episodes sampled during training (default: 50)
```

The agent can use a variety of network architectures, available in `rl_traffic_controller/networks.py`, one of them must be selected in the command line. The list of available architectures can be viewed in the help message, `python run.py -h`. The commands below use the `v3` architecture.

To train the agent from scratch, run the following command and replace `N` with the number of episodes you want.

```bash
python run.py train v3 --episodes N
```

To continue training the agent using a previously saved Q network, run the following command.

```bash
python run.py train v3 --episodes N --continue
```

To see the agent's action values (using a previously saved Q network), run the following command and provide as many pictures, and directories containing multiple pictures, as you want. The directories will be searched for pictures in the first level only.

```bash
python run.py eval v3 firstPicture.png secondPicture.png pictureDir/ ...
```

To see the agent's action values after the training is finished, provide the pictures and/or directories when you start the training.

```bash
python run.py train v3 firstPicture.png secondPicture.png pictureDir/ ... --episodes N
```

To demo the agent in a running environment use the following command.

```bash
python run.py demo v3
```

### Notes

- The agent automatically saves the Q network after each episode in the `models/` directory.
- The `logs/run.log` provides extensive logs during the training process.
- During training, it's preferable to keep the simulation window visible, as it will make capturing screenshots noticably faster.
- Network configurations depend on the size of the input image, thus some configurations may not work with the current image size.
