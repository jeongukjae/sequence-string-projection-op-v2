import tensorflow as tf
import sequence_string_projection_op_v2 as seq_proj


class SequenceStringProjectionV2Test(tf.test.TestCase):
    def testOutput(self):
        with self.session():
            # test initialize
            seq_proj.SequenceStringProjectionV2(
                input=tf.keras.Input([], dtype=tf.string),
                sequence_length=tf.keras.Input([], dtype=tf.int32),
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
            )

            with self.assertRaises(tf.python.framework.errors_impl.InvalidArgumentError):
                # `input` must be a matrix, got shape: [2,8,1]
                seq_proj.SequenceStringProjectionV2(
                    input=tf.reshape(tf.constant(["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]), [2, 8, 1]),
                    sequence_length=tf.constant([9, 0, 9]),
                    feature_size=16,
                )

            with self.assertRaises(tf.python.framework.errors_impl.InvalidArgumentError):
                # `sequence_length` must be a vector, got shape: [3,1]
                seq_proj.SequenceStringProjectionV2(
                    input=tf.reshape(tf.constant(["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]), [2, 8]),
                    sequence_length=tf.constant([[9, 0, 9]]),
                    feature_size=16,
                )


            with self.assertRaises(tf.python.framework.errors_impl.InvalidArgumentError):
                # `sequence_length` should have batch size number of elements, got size 3, batch size is 2
                seq_proj.SequenceStringProjectionV2(
                    input=tf.reshape(tf.constant(["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]), [2, 8]),
                    sequence_length=tf.constant([9, 0, 9]),
                    feature_size=16,
                )

            with self.assertRaises(tf.python.framework.errors_impl.InvalidArgumentError):
                # `sequence_length` should have values less than or equal to max_seq_len
                seq_proj.SequenceStringProjectionV2(
                    input=tf.reshape(tf.constant(["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]), [2, 8]),
                    sequence_length=tf.constant([9, -1]),
                    feature_size=16,
                )

            with self.assertRaises(tf.python.framework.errors_impl.InvalidArgumentError):
                # `sequence_length` should have values greater than or equal to 0
                seq_proj.SequenceStringProjectionV2(
                    input=tf.reshape(tf.constant(["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij", "quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]), [2, 8]),
                    sequence_length=tf.constant([4, -1]),
                    feature_size=16,
                )

            # OK
            result = seq_proj.SequenceStringProjectionV2(
                input=tf.constant([["hello", "world", "147", "dog", "xyz", "abc", "efg", "hij"], ["quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]]),
                sequence_length=tf.constant([4, 8]),
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
            )

            self.assertAllEqual(result.shape, [2, 8, 16])

            self.assertNotAllEqual(result[0, 0], result[1, 0]) # hello != quick.
            self.assertNotAllEqual(result[0, 1], result[1, 1]) # world != hello.
            self.assertAllEqual(result[0, 0], result[1, 1]) # hello == hel1lo.
            self.assertAllEqual(result[0, 2], result[1, 2]) # 147 == 123 (oov values).
            self.assertAllEqual(result[0, 3], result[1, 7]) # dog == dog.

            for i in range(4, 8):
                self.assertAllEqual(result[0, i], tf.zeros_like(result[0, i]))

    def testOutputBoS(self):
        with self.session():
            input_tensor = tf.constant([["hello", "world", "147", "dog", "", "", "", ""], ["quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]])
            sequence_length_tensor = tf.constant([4, 8])

            result = seq_proj.SequenceStringProjectionV2(
                input=input_tensor,
                sequence_length=sequence_length_tensor,
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
                add_bos_tag=True
            )

            self.assertAllEqual(result.shape, [2, 9, 16])

            self.assertAllEqual(result[0, 0], result[1, 0]) # <bos> == <bos>.
            self.assertNotAllEqual(result[0, 1], result[1, 1]) # hello != quick.
            self.assertNotAllEqual(result[0, 2], result[1, 2]) # world != hello.
            self.assertAllEqual(result[0, 1], result[1, 2]) # hello == hel1lo.
            self.assertAllEqual(result[0, 3], result[1, 3]) # 147 == 123 (oov values).
            self.assertAllEqual(result[0, 4], result[1, 8]) # dog == dog.

            for i in range(5, 9):
                self.assertAllEqual(result[0, i], tf.zeros_like(result[0, i]))


    def testOutputEoS(self):
        with self.session():
            input_tensor = tf.constant([["hello", "world", "147", "dog", "", "", "", ""], ["quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]])
            sequence_length_tensor = tf.constant([4, 8])

            result = seq_proj.SequenceStringProjectionV2(
                input=input_tensor,
                sequence_length=sequence_length_tensor,
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
                add_eos_tag=True
            )

            self.assertAllEqual(result.shape, [2, 9, 16])

            self.assertNotAllEqual(result[0, 0], result[1, 0]) # hello != quick.
            self.assertNotAllEqual(result[0, 1], result[1, 1]) # world != hello.
            self.assertAllEqual(result[0, 0], result[1, 1]) # hello == hel1lo.
            self.assertAllEqual(result[0, 2], result[1, 2]) # 147 == 123 (oov values).
            self.assertAllEqual(result[0, 3], result[1, 7]) # dog == dog.
            self.assertAllEqual(result[0, 4], result[1, 8]) # <bos> == <bos>.

            for i in range(5, 9):
                self.assertAllEqual(result[0, i], tf.zeros_like(result[0, i]))

    def testOutputBoSEoS(self):
        with self.session():
            input_tensor = tf.constant([["hello", "world", "147", "dog", "...", "..", "", ""], ["quick", "hel1lo", "123", "jumped", "over", "the", "lazy", "dog"]])
            sequence_length_tensor = tf.constant([6, 8])

            result = seq_proj.SequenceStringProjectionV2(
                input=input_tensor,
                sequence_length=sequence_length_tensor,
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
                add_bos_tag=True,
                add_eos_tag=True
            )

            self.assertAllEqual(result.shape, [2, 10, 16])

            self.assertAllEqual(result[0, 0], result[1, 0]) # <bos> == <bos>.
            self.assertNotAllEqual(result[0, 1], result[1, 1]) # hello != quick.
            self.assertNotAllEqual(result[0, 2], result[1, 2]) # world != hello.
            self.assertAllEqual(result[0, 1], result[1, 2]) # hello == hel1lo.
            self.assertAllEqual(result[0, 3], result[1, 3]) # 147 == 123 (oov values).
            self.assertAllEqual(result[0, 4], result[1, 8]) # dog == dog.
            self.assertAllEqual(result[0, 7], result[1, 9]) # <eos> == <eos>.
            self.assertNotAllEqual(result[0, 4], result[0, 5]) # ... != ..

            for i in range(8, 10):
                self.assertAllEqual(result[0, i], tf.zeros_like(result[0, i]))

    def testOutputNormalize(self):
        with self.session():
            input_tensor = tf.constant([["hello", "world", "..", "....", "", "", "", ""], ["quick", "hel1lo", "123", "jumped", "over", "...", ".....", "dog"]])
            sequence_length_tensor = tf.constant([4, 8])

            result = seq_proj.SequenceStringProjectionV2(
                input=input_tensor,
                sequence_length=sequence_length_tensor,
                normalize_repetition=True,
                feature_size=16,
                vocabulary='abcdefghijklmnopqrstuvwxyz',
            )

            self.assertAllEqual(result.shape, [2, 8, 16])

            self.assertAllEqual(result[0, 2], result[0, 3]) # .. == ....
            self.assertAllEqual(result[1, 5], result[1, 6]) # ... == ..
            self.assertAllEqual(result[0, 3], result[1, 6]) # .... == ...

            for i in range(4, 8):
                self.assertAllEqual(result[0, i], tf.zeros_like(result[0, i]))

if __name__ == "__main__":
      tf.test.main()
