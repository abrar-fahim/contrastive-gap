import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# add sibling directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clips.hf_clip import HFClip

from src.utils import evaluate_linearity


# load clip model
clip_model = HFClip()

evaluate_linearity(clip_model, plot=True)

