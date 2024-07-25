# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python main.py --workdir=/tmp/mnist --config=configs/default.py
# python main.py --workdir=. --config=configs/default.py


"""Main file for running the MNIST example.

This file is intentionally kept short. The majority of logic is in libraries
than can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf

import train


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")  # --workdir
config_flags.DEFINE_config_file(  # --config
    "config",  # name
    None,  # default
    "File path to the training hyperparameter configuration.",  # help_string
    # lock_config – If set to True, loaded config will be locked through calling .lock() method on its instance (if it exists). (default: True)
    # TODO
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    # Set the list of visible devices.
    tf.config.experimental.set_visible_devices([], "GPU")

    # INFO:abs:...
    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        # process_index : Returns the integer process index of this process. On most platforms, this will always be 0. This will vary on multi-process platforms though.
        # process_count : Returns the total number of devices.
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )

    # work_unit() -> LocalWorkUnit(WorkUnit)
    #   create_artifact() -> Artifact(WorkUnit)
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir")

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    # FATAL Flags parsing error:
    #   flag --config=None: Flag --config must have a value other than None.
    #   flag --workdir=None: Flag --workdir must have a value other than None.
    # Pass --helpshort or --helpfull to see help on flags.
    app.run(main)
