"""
DTLN (Dual-signal Transformation LSTM Network) model for noise reduction.

This is a self-contained, inference-only version of the DTLN model architecture.
Original paper: https://arxiv.org/abs/2005.07551
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation, Dense, LSTM, Dropout,
    Lambda, Input, Multiply, Layer, Conv1D
)


class InstantLayerNormalization(Layer):
    """
    Instant (channel-wise) layer normalization.
    
    Proposed by Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, **kwargs):
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        shape = input_shape[-1:]
        self.gamma = self.add_weight(
            shape=shape, initializer='ones', trainable=True, name='gamma'
        )
        self.beta = self.add_weight(
            shape=shape, initializer='zeros', trainable=True, name='beta'
        )

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(
            tf.math.square(inputs - mean), axis=[-1], keepdims=True
        )
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma
        outputs = outputs + self.beta
        return outputs


class DTLN_model:
    """
    Dual-signal Transformation LSTM Network for noise reduction.
    
    This class builds the DTLN model architecture and loads pre-trained weights
    for inference. It does not include training utilities.
    """

    def __init__(self):
        self.model = None
        # Model hyperparameters (must match training configuration)
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.encoder_size = 256
        self.eps = 1e-7

    # ---- Signal processing layers ----

    def stftLayer(self, x):
        """STFT: returns [magnitude, phase]"""
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        stft_dat = tf.signal.rfft(frames)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return [mag, phase]

    def ifftLayer(self, x):
        """Inverse FFT from [magnitude, phase] to time-domain frames."""
        s1_stft = (
            tf.cast(x[0], tf.complex64)
            * tf.exp(1j * tf.cast(x[1], tf.complex64))
        )
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        """Overlap-and-add to reconstruct waveform from frames."""
        return tf.signal.overlap_and_add(x, self.block_shift)

    # ---- Core model components ----

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        """Separation kernel: stacked LSTMs followed by a Dense + Activation mask."""
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            if idx < (num_layer - 1):
                x = Dropout(self.dropout)(x)
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        return mask

    # ---- Model builders ----

    def build_DTLN_model(self, norm_stft=False):
        """
        Build the DTLN model for batch inference.
        
        Args:
            norm_stft: If True, apply log-magnitude normalization (must match
                       the setting used during training).
        """
        # Input: time-domain waveform
        time_dat = Input(batch_shape=(None, None))

        # STFT
        mag, angle = Lambda(self.stftLayer)(time_dat)

        # Optional log-magnitude normalization
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            mag_norm = mag

        # First separation core (frequency domain)
        mask_1 = self.seperation_kernel(
            self.numLayer, (self.blockLen // 2 + 1), mag_norm
        )
        estimated_mag = Multiply()([mag, mask_1])
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])

        # Learned encoder
        encoded_frames = Conv1D(
            self.encoder_size, 1, strides=1, use_bias=False
        )(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)

        # Second separation core (learned feature domain)
        mask_2 = self.seperation_kernel(
            self.numLayer, self.encoder_size, encoded_frames_norm
        )
        estimated = Multiply()([encoded_frames, mask_2])

        # Decoder back to time domain
        decoded_frames = Conv1D(
            self.blockLen, 1, padding='causal', use_bias=False
        )(estimated)
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)

        self.model = Model(inputs=time_dat, outputs=estimated_sig)
