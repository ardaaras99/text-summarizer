from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    from textSummarizer.logging import logger
    from pathlib import Path

    print(Path.cwd())

    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f"Running {STAGE_NAME}...")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f"Completed {STAGE_NAME}.")
    except Exception as e:
        logger.error(f"Failed to run {STAGE_NAME}. Error: {e}")
        raise e
