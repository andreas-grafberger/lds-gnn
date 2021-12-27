import logging
from os import getenv
from typing import List

from sacred import Experiment
from sacred.observers import TelegramObserver
from torch.optim.optimizer import Optimizer


def setup_basic_logger():
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


def attach_telegram_observer(ex: Experiment):
    """
    Adds support to automatically send latest experiment results via telegram. 
    Requires that the TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_CHAT_ID environment variables are set.
    """
    try:
        import telegram
        telegram_token = getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = getenv("TELEGRAM_BOT_CHAT_ID")
        if telegram is not None and telegram_token is not None and telegram_chat_id is not None:
            ex.observers.append(TelegramObserver(telegram.Bot(telegram_token), chat_id=telegram_chat_id))
    except ImportError:
        ex.logger.warn("Could not import telegram, make sure installation is valid.")


def attach_neptune_observer(ex: Experiment, project_name: str = None):
    """
    Adds neptune.ml support to automatically upload experiment results to their platform.
    Requires that the NEPTUNE_API_TOKEN environment variable is set.
    :param ex: Experiment instance
    :param project_name: Name of the current project. Has to match the one on the neptune.ml website. If not specified,
    no observer is added.
    """
    try:
        from neptunecontrib.monitoring.sacred import NeptuneObserver
        project_name = project_name or getenv("NEPTUNE_PROJECT_NAME")
        if project_name is not None:
            ex.observers.append(NeptuneObserver(project_name=project_name))
    except ImportError:
        ex.logger.warn("Could not import neptune observer. Run 'pip install neptune-contrib' first.")


# noinspection Mypy
def get_lr(optimizer: Optimizer) -> List[float]:
    return [group['lr'] for group in optimizer.param_groups]
