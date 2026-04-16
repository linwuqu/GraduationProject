from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import callbacks, layers


@dataclass
class SDAEConfig:
    layer_units: tuple[int, ...] = (24, 12, 6)
    dropout_rate: float = 0.1
    batch_size: int = 64
    epochs: int = 40
    validation_split: float = 0.15
    random_state: int = 42
    one_class_label: int | None = None


class SDAEFeatureExtractor:
    def __init__(self, config: SDAEConfig | None = None) -> None:
        self.config = config or SDAEConfig()
        self.autoencoders: list[keras.Model] = []
        self.encoders: list[keras.Model] = []
        self.input_dim: int | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> None:
        train_features = features.to_numpy(dtype=np.float32)
        if self.config.one_class_label is not None and target is not None:
            mask = target.to_numpy() == self.config.one_class_label
            if mask.any():
                train_features = train_features[mask]

        self.input_dim = train_features.shape[1]
        current = train_features
        self.autoencoders = []
        self.encoders = []
        keras.utils.set_random_seed(self.config.random_state)

        for units in self.config.layer_units:
            autoencoder, encoder = self._build_single_layer_model(current.shape[1], units)
            stopper = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
            autoencoder.fit(
                current,
                current,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                shuffle=True,
                verbose=0,
                callbacks=[stopper],
            )
            current = encoder.predict(current, verbose=0)
            self.autoencoders.append(autoencoder)
            self.encoders.append(encoder)

    def transform(self, features: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        if not self.encoders:
            raise RuntimeError("SDAEFeatureExtractor must be fitted before transform.")

        current = features.to_numpy(dtype=np.float32)
        layer_features: list[pd.DataFrame] = []
        for layer_idx, encoder in enumerate(self.encoders, start=1):
            current = encoder.predict(current, verbose=0)
            columns = [f"sdae_l{layer_idx}_{idx}" for idx in range(current.shape[1])]
            layer_features.append(pd.DataFrame(current, index=features.index, columns=columns))

        reconstruction = self.autoencoders[0].predict(features.to_numpy(dtype=np.float32), verbose=0)
        reconstruction_error = np.mean((features.to_numpy(dtype=np.float32) - reconstruction) ** 2, axis=1)
        reconstruction_frame = pd.DataFrame(
            {"sdae_reconstruction_error": reconstruction_error},
            index=features.index,
        )
        all_features = pd.concat([features, *layer_features, reconstruction_frame], axis=1)
        return all_features, layer_features

    def fit_transform(self, features: pd.DataFrame, target: pd.Series | None = None) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        self.fit(features, target=target)
        return self.transform(features)

    def _build_single_layer_model(self, input_dim: int, units: int) -> tuple[keras.Model, keras.Model]:
        inputs = layers.Input(shape=(input_dim,))
        noisy = layers.Dropout(self.config.dropout_rate)(inputs)
        encoded = layers.Dense(
            units,
            activation="selu",
            activity_regularizer=keras.regularizers.l1(1e-5),
        )(noisy)
        decoded = layers.Dense(input_dim, activation="linear")(encoded)
        autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        encoder = keras.Model(inputs=inputs, outputs=encoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder, encoder
