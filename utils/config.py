import os


class Config:
    BASE_DIR = ""
    DATA_DIR = os.path.join(BASE_DIR, "data")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
