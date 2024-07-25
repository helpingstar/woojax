from absl import app
from ml_collections import config_flags

"""
import ml_collections


def get_config():
  # Get the default hyperparameter configuration
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 10
  return config
"""

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    print(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)
