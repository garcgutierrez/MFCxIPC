import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(RES, "models")
FIG = os.path.join(ROOT, "paper", "figures")
sys.path.insert(0, CODE)
os.makedirs(MODELS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
