from absl import app, flags
from ml_collections import config_dict
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.learning_rate = 0.01
    cfg.batch_size = 32
    return cfg


# Define the config flag with lock_config set to True (default)
config_flags.DEFINE_config_dict(
    "config", get_config(), "ConfigDict instance.", lock_config=True  # This will lock the config
)


def main(argv):
    # Access the configuration
    config = FLAGS.config
    print("hello")
    # Attempting to modify the config will raise an error if lock_config is True
    try:
        config.learning_rate = 0.02
    except Exception as e:
        print(f"Failed to modify config: {e}")


if __name__ == "__main__":
    app.run(main)
