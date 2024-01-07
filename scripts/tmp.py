import sys
import os

EPS_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(os.path.dirname(EPS_DIR))
from EPS.src.nodes import PWS
print('success')
