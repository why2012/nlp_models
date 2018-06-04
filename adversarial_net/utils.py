import logging

def getLogger(name=None):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    return logger