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
from tictactoe_bot import TicTacToeBot
from utils import setup_tee_logging


# The draccus decorator handles command-line argument parsing
@draccus.wrap()
def main(cfg: TicTacToeConfig):
    setup_tee_logging()
    init_logging()
    logging.info(pformat(asdict(cfg)))
    # """Main function to run the Tic-Tac-Toe demo."""
    ttt_bot = TicTacToeBot(cfg)
    ttt_bot.run()


if __name__ == "__main__":
    main()
