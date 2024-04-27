"""A module to hold constants."""

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
LR = 0.0001
MEMORY_SIZE = 2000

SIMULATION_FLOW_PROBABILITY = (0.075, 0.005)
SIMULATION_PHASES = [
    "GGgGGrrrrrrrGGgGGrrrrrrr",
    "rrrrrGrrrrrrrrrrrGrrrrrr",
    "rrrrrrGGgGGrrrrrrrGGgGGr",
    "rrrrrrrrrrrGrrrrrrrrrrrG",
]
SIMULATION_AMBER_PHASES = [
    "yyyyyrrrrrrryyyyyrrrrrrr",
    "rrrrryrrrrrrrrrrryrrrrrr",
    "rrrrrryyyyyrrrrrrryyyyyr",
    "rrrrrrrrrrryrrrrrrrrrrry"
]
SIMULATION_EDGE_IDS = ["E2TL", "N2TL", "S2TL", "W2TL"]
SIMULATION_TRAFFIC_LIGHT_ID = "TL"
SIMULATION_ROUTE_PATH = r"simulation/v1.rou.xml"
SIMULATION_CONFIG_PATH = r"./simulation/v1.sumocfg"

IMAGE_SIZE = (200, 133)
IMAGE_FORMAT = "L"
IMAGE_PATH = r"data/simulation.png"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
