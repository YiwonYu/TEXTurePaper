import time
import logging
import torch
import os
import pyrallis
import sys

# --- Option 1: Configure logging using basicConfig (Simpler, often robust) ---
# Do this *before* importing other modules that might configure logging
# Use force=True (Python 3.8+) to override any other potential basicConfig calls
log_file_path_basic = 'run_basic.log'
try:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", # Added logger name
        handlers=[
            logging.FileHandler(log_file_path_basic, mode='a'),
            logging.StreamHandler(sys.stdout) # Also log to console
        ],
        force=True # Uncomment if using Python 3.8+ and suspecting conflicts
    )
    logging.info(f"Logging initialized via basicConfig to {log_file_path_basic}")
except IOError as e:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.error(f"Failed to open log file {log_file_path_basic} due to {e}. Logging to console only.")

# Get a logger instance (can be the root logger or a specific one)
# Using a specific name is still good practice if TEXTure uses its own logger
logger = logging.getLogger('texture_app') # Use a descriptive name

# --- Option 2: Keep your detailed setup (place it strategically) ---
# If you prefer your original setup, ensure it runs without interference.
# It's generally better to configure logging once. If using basicConfig above,
# you might comment out the setup inside main(). If using the setup below,
# comment out the basicConfig above.

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure

@pyrallis.wrap()
def main(cfg: TrainConfig):
    # --- Start detailed logger setup (if not using basicConfig above) ---
    # logger = logging.getLogger('my_logger') # Use the logger instance from basicConfig or define here
    # logger.setLevel(logging.INFO)
    # if not logger.handlers: # Check to prevent adding handlers multiple times
    #     try:
    #         log_file_path = 'run_detailed.log'
    #         print(f"CWD: {os.getcwd()}") # Debug print CWD
    #         print(f"Attempting to create log file at: {os.path.join(os.getcwd(), log_file_path)}") # Debug print
    #
    #         fh = logging.FileHandler(log_file_path, mode='a')
    #         fh.setLevel(logging.INFO)
    #         formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    #         fh.setFormatter(formatter)
    #         logger.addHandler(fh)
    #
    #         # Optional: Add console handler too for visibility
    #         ch = logging.StreamHandler(sys.stdout)
    #         ch.setLevel(logging.INFO)
    #         ch.setFormatter(formatter)
    #         logger.addHandler(ch)
    #
    #         logger.info("Dedicated logger initialized.")
    #
    #     except IOError as e:
    #         print(f"CRITICAL: Could not open log file '{log_file_path}'. Error: {e}. Check permissions/path.")
    #         # Fallback to console only if file fails
    #         if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    #              ch = logging.StreamHandler(sys.stdout)
    #              ch.setLevel(logging.INFO)
    #              ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    #              logger.addHandler(ch)
    #              logger.error(f"File logging failed: {e}. Using console only.")
    # --- End detailed logger setup ---


    try: # Wrap core logic in try...except
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info("Starting training/evaluation.")

        # Log the YAML config file name.
        config_path_logged = False
        if hasattr(cfg, "config_path") and cfg.config_path:
            logger.info(f"Running config file (from cfg): {cfg.config_path}")
            config_path_logged = True
        else:
            for arg in sys.argv:
                if isinstance(arg, str) and arg.endswith('.yaml'):
                    logger.info(f"Running config file (from argv): {arg}")
                    config_path_logged = True
                    break
        if not config_path_logged:
            logger.warning("Could not determine config file path to log.")

        start_time = time.time()

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"GPU memory allocated before: {allocated_gb:.2f} GB")
            logger.info(f"GPU memory reserved before: {reserved_gb:.2f} GB")

        logger.info("Initializing TEXTure Trainer...")
        trainer = TEXTure(cfg)
        logger.info("TEXTure Trainer Initialized.")

        if cfg.log.eval_only:
            logger.info("Starting evaluation...")
            trainer.full_eval()
            logger.info("Evaluation finished.")
        else:
            logger.info("Starting painting...")
            trainer.paint()
            logger.info("Painting finished.")

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"GPU memory allocated after: {allocated_gb:.2f} GB")
            logger.info(f"GPU memory reserved after: {reserved_gb:.2f} GB")

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info("Script finished successfully.")

    except Exception as e:
        logger.exception("An error occurred during execution:") # Logs exception info
        print(f"An error occurred: {e}", file=sys.stderr) # Also print to stderr
        sys.exit(1) # Indicate failure
    finally:
        # Explicitly shutdown logging (optional, usually happens on exit)
        # logging.shutdown()
        pass

if __name__ == '__main__':
    # Ensure CWD is printed early if basicConfig fails
    print(f"Executing script from: {os.getcwd()}")
    main()