# Mastering BipedalLocomotion via TD3
**Author:** Gregory Li | gregli@g.ucla.edu

> Using the Twin Delayed DDPG (TD3) algorithm to solve the Gymnasium BipedalWalker-v3 environment.

[Champion Walker Video](videos/final_eval.mp4) 

## Quick Start
To replicate my results:
1. `pip install -r requirements.txt`
2. `python src/record.py` (This will use the pre-trained `td3_walker.zip`)

## Repository Structure
```
├── models/
│   ├── td3_walker.zip        # Pre-trained Actor/Critic weights
│   └── vec_normalize.pkl     # Observation mean/std stats
├── src/
│   ├── train_td3.py              # Script used for the 550k step training run
│   └── record_walker.py      # Script to load model and generate champion video
├── videos/
│   └── final_eval.mp4        # The 300+ reward champion run video
├── .gitignore
├── requirements.txt
└── README.md