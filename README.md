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

usage: run.py [-h] [-c] [-s] [-p] [-e N] [-a agent_name] [--images  [...]] mode

positional arguments:
  mode                  train or eval or demo or dry-run

options:
  -h, --help            show this help message and exit
  -c, --continue        load the saved networks and continue training
  -s, --save            save the networks after every training episode
  -p, --plot            plot the metrics after training
  -e N, --episodes N    number of episodes sampled during training (default: 1)
  -a agent_name, --agent agent_name
                        which agent to use, dqn or sac or fixed (default: dqn)
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

The `--agent` option specifies which agent to use. Use `python3.11 run.py --help` to know the available agents and which agent is used by default.

```bash
python3.11 run.py train --agent agent_name --episodes N
```

To plot the metrics collected in the training run, use the option `--plot`.

```bash
python3.11 run.py train --episodes N --plot
```

To see the agent's action values (using a previously saved Q network), run the following command and provide as many images, and directories containing multiple images, as you want. The directories will be searched for images in the first level only. The `--images` option should be the last option used in the command.

```bash
python3.11 run.py eval --images firstPicture.png secondPicture.png imagesDir/ ...
```

To see the agent's action values after the training is finished, provide the images and/or directories when you start the training.

```bash
python3.11 run.py train --episodes N --images firstPicture.png secondPicture.png imagesDir/ ...
```

To demo the agent in a running environment use the following command. You can use the `--episodes` option to specify the number of episodes.

```bash
python3.11 run.py demo --episodes N
```

If you want to test the code without using SUMO, use the `dry-run` mode to simulate the training process. It functions exactly as the `train` mode.

```bash
python3.11 run.py dry-run
```

### Notes

- The agents save any networks and related variables after each episode in the `models/` directory.
- The `logs/run.log` provides extensive logs during the training process.
- During training, it's preferable to keep the simulation window visible, as it will make capturing screenshots noticably faster.
- Network configurations depend on the size and format of the input image, thus some configurations may not work with the current settings (specified in `rl_traffic_controller/consts.py`).
- The `fixed` agent runs a traditional timer-based traffic light controller. It's used as a benchmark for other agents, and can't be used in `eval` mode.
