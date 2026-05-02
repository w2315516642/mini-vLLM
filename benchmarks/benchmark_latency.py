import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from minivllm import LLM, SamplingParams