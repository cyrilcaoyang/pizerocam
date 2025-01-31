
"""
A logger Class that handles logging

"""

import os
import logging
import logging.config
import yaml


class Logger:
    """
    A Logger class that, upon instantiation, creates a Projects/Logs folder
    in the user's home directory, updates all FileHandlers to use that folder,
    and configures logging accordingly.
    """
    def __init__(self, config_filename):
        # Directory containing this logger.py file
        self.config_path = os.path.join(self.script_dir, config_filename)

        # Directory where log files will be stored
        self.logs_dir = os.path.join(os.path.expanduser("~"), "Projects", "Logs")

        # Internal cache of the YAML config
        self.config = None

        # Set up logging as soon as class is instantiated
        self._setup_logging()

    def _setup_logging(self):
        """
        Orchestrates all steps to set up logging from the YAML config.
        """
        self._read_config_file()
        self._create_logs_directory()
        self._update_file_handler_paths()
        self._configure_logging()

    def _read_config_file(self):
        """
        Reads the logger_settings.yaml file into self.config.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Could not find logging config file at: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _create_logs_directory(self):
        """
        Ensures the 'Projects/Logs' directory exists in the user's home directory.
        """
        os.makedirs(self.logs_dir, exist_ok=True)

    def _update_file_handler_paths(self):
        """
        Modifies any FileHandler filenames so logs go into ~/Projects/Logs.
        """
        handlers = self.config.get("handlers", {})
        for handler_name, handler_cfg in handlers.items():
            if handler_cfg.get("class") == "logging.FileHandler":
                original_filename = handler_cfg.get("filename")
                if original_filename:
                    # Prepend the logs directory
                    handler_cfg["filename"] = os.path.join(
                        self.logs_dir,
                        os.path.basename(original_filename)
                    )

    def _configure_logging(self):
        """
        Applies the updated logging configuration via dictConfig.
        """
        logging.config.dictConfig(self.config)

    def get_logger(self, name=None):
        """
        Returns a logger instance. If name is None, returns the root logger.
        """
        return logging.getLogger(name)

# Example usage when you import this class in a different script:
if __name__ == "__main__":
    # Instantiate Logger
    logger_obj = Logger()          # will automatically configure logging
    logger = logger_obj.get_logger("my_logger")

    logger.debug("Debug message from my_logger.")
    logger.info("Info message from my_logger.")