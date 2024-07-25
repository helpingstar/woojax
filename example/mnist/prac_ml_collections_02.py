from ml_collections import config_dict
from ml_collections import config_flags
from absl import app

config = config_dict.ConfigDict(
    {
        "field1": 1,
        "field2": "tom",
        "nested": {
            "field": 2.23,
        },
    }
)


_CONFIG = config_flags.DEFINE_config_dict("my_config", config)


def main(argv):
    print(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)


"""
python prac_ml_collections_02.py --helpshort
python prac_ml_collections_02.py --my_config.field1 5
python prac_ml_collections_02.py --my_config.nested.field 2.55555
"""
