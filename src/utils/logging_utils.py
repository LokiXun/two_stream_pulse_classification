#!/usr/bin/env python3
"""日志工具类"""
# -*- coding:utf-8 -*-
from pathlib import Path
import platform
from typing import Optional
import logging
from logging import handlers
import datetime

import yaml

CONFIG_PASSED = False
CLIENT_CONFIG_PASSED = False
ROOT_LOGGER: Optional[logging.Logger] = None

if platform.system() == "Linux":
    store_parent_dir = Path(__file__).resolve().parent
else:
    # windows
    store_parent_dir = Path(__file__).resolve().parent
store_parent_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name="video_highlight_compilation"):
    """日志client端获取logger对象"""
    global CLIENT_CONFIG_PASSED, ROOT_LOGGER
    if not CLIENT_CONFIG_PASSED:
        ROOT_LOGGER = logging.getLogger(name)
        ROOT_LOGGER.setLevel(logging.INFO)

        rf_handler = logging.handlers.TimedRotatingFileHandler(
            store_parent_dir.joinpath(f"{name}.log").as_posix(),
            when="midnight",
            interval=1,
            backupCount=7,
            atTime=datetime.time(0, 0, 0, 0)
        )
        rf_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)8s %(process)6d %(module)30s:%(lineno)-4d >> %(message)s"))

        ROOT_LOGGER.addHandler(rf_handler)
        CLIENT_CONFIG_PASSED = True
    return ROOT_LOGGER


if __name__ == '__main__':
    logger = get_logger()
    logger.error(f"test logger")
