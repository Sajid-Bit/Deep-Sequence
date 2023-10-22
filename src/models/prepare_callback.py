from src.entity.config_entity import PrepareCallbackConfig
import tensorflow as tf


class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config

    @property
    def create_ckpt_callbacks(self):
        """
        Return Check point will the model training and save the model in the models dir

        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )
