# Standard library
import logging
import os
import sys
from dataclasses import asdict

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402 # add SO-100 to path for imports

from pprint import pformat

# Third-party libraries
import draccus

# Local modules
from config import TicTacToeConfig
from lerobot.utils.utils import init_logging
from robot_controller import TicTacToeBot
from utils import setup_tee_logging
from vlm_client import VLMClient


# The draccus decorator handles command-line argument parsing
@draccus.wrap()
def main(cfg: TicTacToeConfig):
    setup_tee_logging()
    init_logging()
    logging.info(pformat(asdict(cfg)))
    # """Main function to run the Tic-Tac-Toe demo."""
    vlm_client = VLMClient()
    ttt_bot = TicTacToeBot(vlm_client, cfg)
    ttt_bot.run()


if __name__ == "__main__":
    main()
