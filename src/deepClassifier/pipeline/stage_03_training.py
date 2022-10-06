
from deepClassifier.config import ConfigurationManager
from deepClassifier.components import PrepareCallback, Training
from deepClassifier import logger

STAGE_NAME = "Training"

def main():
    try:
        config = ConfigurationManager()
        prepare_callbacks_model_config = config.get_prepare_callback_config()
        prepare_callbacks_model = PrepareCallback(config=prepare_callbacks_model_config)
        callback_list = prepare_callbacks_model.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )
    except Exception as e:
        raise e

if __name__ =='__main__':
    try:
        logger.info("*********************************")
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed<<<<<<\n\nx==============x")
    except Exception as e:
        raise e