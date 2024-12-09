# utils/helpers.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def extract_time(data):
    """
    Extract maximum sequence length and individual sequence lengths.

    Args:
        data (np.ndarray): Original data.

    Returns:
        tuple: List of sequence lengths and the maximum sequence length.
    """
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        seq_len = len(data[i][:, 0])
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)
    return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
    """
    Create an RNN cell based on the specified module.

    Args:
        module_name (str): Type of RNN cell ('gru', 'lstm', 'lstmLN').
        hidden_dim (int): Number of hidden units.

    Returns:
        tf.nn.rnn_cell.RNNCell: Configured RNN cell.
    """
    assert module_name in ["gru", "lstm", "lstmLN"]

    if module_name == "gru":
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
    elif module_name == "lstm":
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    elif module_name == "lstmLN":
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
    return rnn_cell


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """
    Generate random noise vectors for the generator.

    Args:
        batch_size (int): Number of samples.
        z_dim (int): Dimension of the noise vector.
        T_mb (list): List of sequence lengths.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: Generated noise vectors.
    """
    Z_mb = []
    for i in range(batch_size):
        temp_Z = np.random.uniform(0.0, 1.0, [T_mb[i], z_dim])
        temp = np.zeros([max_seq_len, z_dim])
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    """
    Generate a mini-batch of data.

    Args:
        data (np.ndarray): Time-series data.
        time (list): List of sequence lengths.
        batch_size (int): Number of samples in the batch.

    Returns:
        tuple: Mini-batch data and corresponding time information.
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb


def get_angles(pos, i, d_model):
    """
    Compute the angles for positional encoding.

    Args:
        pos (np.ndarray): Positions.
        i (np.ndarray): Dimensions.
        d_model (int): Model dimension.

    Returns:
        np.ndarray: Angle rates.
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    Generate positional encoding.

    Args:
        position (int): Maximum position.
        d_model (int): Model dimension.

    Returns:
        tf.Tensor: Positional encoding matrix.
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class TokenAndPositionEmbedding(layers.Layer):
    """
    Layer to encode position information and add it to the data.
    """
    def __init__(self, maxlen, embed_dim, ff_dim, use_ffn):
        """
        Initialize the layer.

        Args:
            maxlen (int): Maximum sequence length.
            embed_dim (int): Embedding dimension.
            ff_dim (int): Feed-forward network hidden dimension.
            use_ffn (bool): Whether to apply a feed-forward network.
        """
        super(TokenAndPositionEmbedding, self).__init__()
        self.use_ffn = use_ffn
        if self.use_ffn:
            self.ffn = keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ])
        self.pos_emb = positional_encoding(maxlen, embed_dim)

    def call(self, x):
        """
        Apply positional encoding and optional feed-forward network.

        Args:
            x (tf.Tensor): Input data.

        Returns:
            tf.Tensor: Encoded data.
        """
        seq_len = tf.shape(x)[1]
        if self.use_ffn:
            x = self.ffn(x)
        return x + self.pos_emb[:, :seq_len, :]


def create_padding_mask(seq):
    """
    Create a padding mask for the transformer.

    Args:
        seq (tf.Tensor): Input sequence.

    Returns:
        tf.Tensor: Padding mask.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for the transformer decoder.

    Args:
        size (int): Sequence length.

    Returns:
        tf.Tensor: Look-ahead mask.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate scaled dot-product attention.

    Args:
        q (tf.Tensor): Query.
        k (tf.Tensor): Key.
        v (tf.Tensor): Value.
        mask (tf.Tensor): Masking tensor.

    Returns:
        tuple: Output and attention weights.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize the layer.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).

        Args:
            x (tf.Tensor): Input tensor.
            batch_size (int): Batch size.

        Returns:
            tf.Tensor: Split tensor.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        Perform multi-head attention.

        Args:
            v (tf.Tensor): Value tensor.
            k (tf.Tensor): Key tensor.
            q (tf.Tensor): Query tensor.
            mask (tf.Tensor): Masking tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


def point_wise_feed_forward_network(d_model, dff):
    """
    Point-wise feed-forward network.

    Args:
        d_model (int): Model dimension.
        dff (int): Feed-forward network dimension.

    Returns:
        tf.keras.Sequential: Feed-forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer of the transformer.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the encoder layer.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            dff (int): Feed-forward network dimension.
            rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Forward pass for the encoder layer.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Training flag.
            mask (tf.Tensor): Masking tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        attn_output = self.mha(x, x, x, mask)  # (batch_size, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer of the transformer.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initialize the decoder layer.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            dff (int): Feed-forward network dimension.
            rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the decoder layer.

        Args:
            x (tf.Tensor): Input tensor.
            enc_output (tf.Tensor): Encoder output tensor.
            training (bool): Training flag.
            look_ahead_mask (tf.Tensor): Look-ahead mask.
            padding_mask (tf.Tensor): Padding mask.

        Returns:
            tf.Tensor: Output tensor.
        """
        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


def CustomSchedule(step, d_model, warmup_steps=2000):
    """
    Custom learning rate schedule (unused in current implementation).

    Args:
        step (int): Current step.
        d_model (int): Model dimension.
        warmup_steps (int, optional): Number of warmup steps. Defaults to 2000.

    Returns:
        tf.Tensor: Learning rate.
    """
    arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
    arg2 = step * (warmup_steps ** -1.5)
    return tf.math.rsqrt(tf.cast(d_model, tf.float32)) * tf.math.minimum(arg1, arg2)
