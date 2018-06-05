import logging
from collections import defaultdict
import threading
from multiprocessing import Lock

def getLogger(name=None):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    return logger

class LocalVariable(object):
    def __init__(self, default = 0):
        self.variable_dict = defaultdict(lambda : default)

    @property
    def tid(self):
        return threading.get_ident()

    @property
    def value(self):
        return self.variable_dict[self.tid]

    @value.setter
    def value(self, v):
        self.variable_dict[self.tid] = v

    def __add__(self, other):
        return self.variable_dict[self.tid] + other

    def __radd__(self, other):
        return other + self.variable_dict[self.tid]

    def __neg__(self):
        return -self.variable_dict[self.tid]

    def plus(self, plus_v):
        self.variable_dict[self.tid] = self.variable_dict[self.tid] + plus_v

    def minus(self, minus_v):
        self.variable_dict[self.tid] = self.variable_dict[self.tid] - minus_v

    def inverse(self):
        self.variable_dict[self.tid] = -self.variable_dict[self.tid]

class MutexVariable(object):
    def __init__(self, value, name = None):
        self._value = value
        self.name = name
        self.debug = False
        self.lock = Lock()

    @property
    def tid(self):
        return threading.get_ident()

    def __add__(self, other):
        return self._value + other

    def __radd__(self, other):
        return other + self._value

    def __neg__(self):
        return -self._value

    def acquire(self):
        if self.debug:
            print("-----id-%s-" % self.tid, self.name, "acquire")
        self.lock.acquire()

    def release(self):
        if self.debug:
            print("-----id-%s-" % self.tid, self.name, "release")
        self.lock.release()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v