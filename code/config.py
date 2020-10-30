import os
from easydict import EasyDict

config = EasyDict({
    "DIRS": {
        "INPUTS": "./data",
        "OUTPUT_MODELS": "./output_models"
    },
    "PATH_PEOPLE": "./people.json",
    "PATH_TAXONOMY": "./label_hierarchy.json",
    "RESOLUTION": {
        "IMAGE": (480, 640, 4),
    },
    "TRAINING": {
        "EPOCHS": 100,
        "DATA_SPLIT_RATIO": 0.8,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 0.5e-3,
        "AUGMENT": True
    }
})
