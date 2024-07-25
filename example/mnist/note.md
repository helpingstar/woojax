```
I0725 16:18:54.720888 125689722461056 main.py:57] JAX process: 0 / 1
I0725 16:18:54.720938 125689722461056 main.py:58] JAX local devices: [cuda(id=0)]
I0725 16:18:54.721075 125689722461056 local.py:45] Setting task status: process_index: 0, process_count: 1
I0725 16:18:54.721148 125689722461056 local.py:50] Created artifact workdir of type ArtifactType.DIRECTORY and value /tmp/mnist.
I0725 16:18:54.875381 125689722461056 dataset_info.py:617] Load dataset info from /home/hs/tensorflow_datasets/mnist/3.0.1
I0725 16:18:54.876659 125689722461056 dataset_info.py:709] For 'mnist/3.0.1': fields info.[citation, splits, supervised_keys, module_name] differ on disk and in the code. Keeping the one from code.
I0725 16:18:54.876790 125689722461056 dataset_builder.py:579] Reusing dataset mnist (/home/hs/tensorflow_datasets/mnist/3.0.1)
I0725 16:18:54.877275 125689722461056 reader.py:261] Creating a tf.data.Dataset reading 1 files located in folders: /home/hs/tensorflow_datasets/mnist/3.0.1.
I0725 16:18:55.296549 125689722461056 logging_logger.py:49] Constructing tf.data.Dataset mnist for split train, from /home/hs/tensorflow_datasets/mnist/3.0.1
I0725 16:18:55.297315 125689722461056 reader.py:261] Creating a tf.data.Dataset reading 1 files located in folders: /home/hs/tensorflow_datasets/mnist/3.0.1.
I0725 16:18:55.399159 125689722461056 logging_logger.py:49] Constructing tf.data.Dataset mnist for split test, from /home/hs/tensorflow_datasets/mnist/3.0.1
I0725 16:19:05.919283 125689722461056 train.py:148] epoch:  1, train_loss: 0.2584, train_accuracy: 92.28, test_loss: 0.0545, test_accuracy: 98.22
I0725 16:19:06.421653 125689722461056 train.py:148] epoch:  2, train_loss: 0.0527, train_accuracy: 98.36, test_loss: 0.0419, test_accuracy: 98.65
I0725 16:19:06.907690 125689722461056 train.py:148] epoch:  3, train_loss: 0.0359, train_accuracy: 98.88, test_loss: 0.0285, test_accuracy: 98.99
I0725 16:19:07.460505 125689722461056 train.py:148] epoch:  4, train_loss: 0.0269, train_accuracy: 99.16, test_loss: 0.0289, test_accuracy: 99.10
I0725 16:19:07.947843 125689722461056 train.py:148] epoch:  5, train_loss: 0.0219, train_accuracy: 99.32, test_loss: 0.0310, test_accuracy: 98.99
I0725 16:19:08.420715 125689722461056 train.py:148] epoch:  6, train_loss: 0.0176, train_accuracy: 99.43, test_loss: 0.0336, test_accuracy: 98.88
I0725 16:19:08.905603 125689722461056 train.py:148] epoch:  7, train_loss: 0.0150, train_accuracy: 99.50, test_loss: 0.0325, test_accuracy: 99.05
I0725 16:19:09.370452 125689722461056 train.py:148] epoch:  8, train_loss: 0.0131, train_accuracy: 99.58, test_loss: 0.0292, test_accuracy: 99.04
I0725 16:19:09.839504 125689722461056 train.py:148] epoch:  9, train_loss: 0.0094, train_accuracy: 99.70, test_loss: 0.0300, test_accuracy: 99.09
I0725 16:19:10.315041 125689722461056 train.py:148] epoch: 10, train_loss: 0.0075, train_accuracy: 99.76, test_loss: 0.0340, test_accuracy: 99.12
```