# `sequence_string_projection_op_v2`

This repository contains minimum source codes to build `sequence_string_projection_op_v2` for tensorflow 2.0 extracted from [`tensorflow/models/research/seq_flow_lite`](https://github.com/tensorflow/models/tree/master/research/seq_flow_lite). The code has been slightly modified so that it can be built on MacOS.

`sequence_string_projection_op_v2` is used in [PRADO](https://www.aclweb.org/anthology/D19-1506.pdf), [pQRNN](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html) to replace the embedding table (tf.keras.layers.Embedding) by hashing input string sequences to ternary vector.

For more details, check out [this link(PRADO)](https://www.aclweb.org/anthology/D19-1506.pdf), and [this link(pQRNN)](https://ai.googleblog.com/2020/09/advancing-nlp-with-efficient-projection.html).

## How to build

You have to install tensorflow and bazel to build this op.

```sh
(env) $ pip install tensorflow
...
(env) $ bazel run //tf_ops:move_ops
...
(env) $ python -c 'from tf_ops import sequence_string_projection_op_v2 as seq_proj; print(seq_proj.SequenceStringProjectionV2.__doc__)'
This op referred to as Ternary Sequence String Projection Op V2 (TSPV2),

  works with presegmented string `input`. It fingerprints each token using murmur
  hash and extracts bit features from the fingerprint that maps every 2 bits to
  the ternary output {-1, 0, 1}. This effectively turns a batch of text segments
  into a ternary rank 3 tensor (in float format) of shape
  [batch size, max sequence length, requested number of features].

  Input(s):
  - input: A string tensor with [batch size, max sequence length] tokens.
  - sequence_length: A vector with batch size number of integers, where each
      integer is in (0, max sequence length], and represents the number of valid
      text segments in each batch entry.

  Attribute(s):
  - feature_size: Length of the ternary vector generated for each token.
  - vocabulary: When not empty provides a list of unique unicode characters that
      will be allowed in the input text before fingerprinting. Expressed another
      way the vocabulary is an optional character allowlist for the
      input tokens. It helps normalize the text.
  - hashtype: Hashing method to use for projection.
  - add_bos_tag: When true inserts a begin of sentence tag.
  - add_eos_tag: When true inserts a end of sentence tag.
  - normalize_repetition: When true normalizes repetition in text tokens before
      fingerprinting.

  Output(s):
  - projection: Floating point tensor with ternary values of shape
      [batch size, max sequence length, requested number of features].

  Args:
    input: A `Tensor` of type `string`.
    sequence_length: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    feature_size: An `int`.
    vocabulary: An optional `string`. Defaults to `""`.
    hashtype: An optional `string`. Defaults to `"murmur"`.
    add_bos_tag: An optional `bool`. Defaults to `False`.
    add_eos_tag: An optional `bool`. Defaults to `False`.
    normalize_repetition: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
(env) $ python tf_ops/sequence_string_projection_op_v2_test.py # run test code
Running tests under Python 3.8.6: /// PYTHON_BIN_PATH ///
[ RUN      ] SequenceStringProjectionV2Test.testOutput
2021-01-16 16:04:29.618379: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-01-16 16:04:29.618560: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO:tensorflow:time(__main__.SequenceStringProjectionV2Test.testOutput): 0.02s
I0116 16:04:29.633468 4727721408 test_util.py:2075] time(__main__.SequenceStringProjectionV2Test.testOutput): 0.02s
[       OK ] SequenceStringProjectionV2Test.testOutput
[ RUN      ] SequenceStringProjectionV2Test.testOutputBoS
INFO:tensorflow:time(__main__.SequenceStringProjectionV2Test.testOutputBoS): 0.01s
I0116 16:04:29.642517 4727721408 test_util.py:2075] time(__main__.SequenceStringProjectionV2Test.testOutputBoS): 0.01s
[       OK ] SequenceStringProjectionV2Test.testOutputBoS
[ RUN      ] SequenceStringProjectionV2Test.testOutputBoSEoS
INFO:tensorflow:time(__main__.SequenceStringProjectionV2Test.testOutputBoSEoS): 0.01s
I0116 16:04:29.653501 4727721408 test_util.py:2075] time(__main__.SequenceStringProjectionV2Test.testOutputBoSEoS): 0.01s
[       OK ] SequenceStringProjectionV2Test.testOutputBoSEoS
[ RUN      ] SequenceStringProjectionV2Test.testOutputEoS
INFO:tensorflow:time(__main__.SequenceStringProjectionV2Test.testOutputEoS): 0.01s
I0116 16:04:29.663685 4727721408 test_util.py:2075] time(__main__.SequenceStringProjectionV2Test.testOutputEoS): 0.01s
[       OK ] SequenceStringProjectionV2Test.testOutputEoS
[ RUN      ] SequenceStringProjectionV2Test.testOutputNormalize
INFO:tensorflow:time(__main__.SequenceStringProjectionV2Test.testOutputNormalize): 0.0s
I0116 16:04:29.668666 4727721408 test_util.py:2075] time(__main__.SequenceStringProjectionV2Test.testOutputNormalize): 0.0s
[       OK ] SequenceStringProjectionV2Test.testOutputNormalize
[ RUN      ] SequenceStringProjectionV2Test.test_session
[  SKIPPED ] SequenceStringProjectionV2Test.test_session
----------------------------------------------------------------------
Ran 6 tests in 0.057s

OK (skipped=1)
```

## How to use

Check out [test cases](https://github.com/jeongukjae/seq_proj_lite/blob/main/tf_ops/sequence_string_projection_op_v2_test.py). I rewrite C++ test code to Python test code for readability.
