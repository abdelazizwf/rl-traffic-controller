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

To train the agent from scratch, run the following command and replace `N` with the number of episodes you want.

```bash
python run.py train --episodes N
```

To continue training the agent using a previously saved Q network, run the following command.

```bash
python run.py train --episodes N --continue
```

To see the agent's action values (using a previously saved Q network), run the following command and provide as many pictures as you want.

```bash
python run.py eval firstPicture.png secondPicture.png ...
```

### Notes

- The agent automatically saves the Q network after each episode in the `models/` directory.
- The `logs/run.log` provides extensive logs during the training process.
- During training, it's preferable to keep the simulation window visible, as it will make capturing screenshots noticably faster.
