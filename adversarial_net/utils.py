import logging
from collections import defaultdict
import threading
from multiprocessing import Lock
import argparse

def getLogger(name=None):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    return logger

class ArgumentsBuilder(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.arguments = None
        self.scope_arguments = defaultdict(list)
        self.name_arguments = []
        self.variables = {}
        self.scope_variables = defaultdict(list)
        self.name_variables = []
        self._registerd_variables = []
        self.scope_associations = defaultdict(list)
        self.name_associations = []
        self.associations = {}
        self.built = False

    def _check_scope_conflict(self, name, scope):
        if scope in self.scope_variables and "%s_%s" % (scope, name) in self.scope_variables[scope]:
            raise Exception("duplicate scope-name: %s_%s" % (scope, name))
        if scope in self.scope_arguments and "%s_%s" % (scope, name) in self.scope_arguments[scope]:
            raise Exception("duplicate scope-name: %s_%s" % (scope, name))
        if scope in self.scope_associations and "%s_%s" % (scope, name) in self.scope_associations[scope]:
            raise Exception("duplicate scope-name: %s_%s" % (scope, name))
        if scope in self.name_variables or scope in self.name_arguments or scope in self.name_associations:
            raise Exception("scope is conflict with variables: %s" % scope)

    def _check_name_conflict(self, name):
        if name in self.name_arguments or name in self.name_variables or name in self.name_associations:
            raise Exception("duplicate variable: %s" % name)
        if name in self.scope_arguments or name in self.scope_variables or name in self.scope_associations:
            raise Exception("name is conflict with scope: %s" % name)

    def str2bool(self, v):
        if v == True:
            return True
        elif v == False:
            return False
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def add_argument(self, name, argtype, scope = None, **kwargs):
        required = False if "default" in kwargs else True
        if scope:
            self._check_scope_conflict(name, scope)
            arg_name = "--{scope}_{name}".format(scope=scope, name=name)
            self.scope_arguments[scope].append("{scope}_{name}".format(scope=scope, name=name))
        else:
            self._check_name_conflict(name)
            arg_name = "--{name}".format(name=name)
            self.name_arguments.append(name)
        if argtype != bool:
            if argtype == "bool":
                self.parser.add_argument(arg_name, type=self.str2bool, required=required, **kwargs)
            else:
                self.parser.add_argument(arg_name, type=argtype, required=required, **kwargs)
        else:
            self.parser.add_argument(arg_name, action="store_true", **kwargs)
        return self

    def add_association(self, name, assoc_name, scope = None, assoc_scope = None):
        if assoc_scope:
            assoc_full_name = [assoc_scope, assoc_name]
        else:
            assoc_full_name = assoc_name
        if scope:
            self._check_scope_conflict(name, scope)
            self.associations["{scope}_{name}".format(scope=scope, name=name)] = assoc_full_name
            self.scope_associations[scope].append("{scope}_{name}".format(scope=scope, name=name))
        else:
            self._check_name_conflict(name)
            self.associations[name] = assoc_full_name
            self.name_associations.append(name)

    def add_scope_association(self, scope, assoc_scope, exclude_names = []):
        for name in self.scope_arguments[assoc_scope]:
            name = name[len(assoc_scope) + 1: ]
            if name not in exclude_names:
                self.add_association(name=name, assoc_name=name, scope=scope, assoc_scope=assoc_scope)
        for name in self.scope_variables[assoc_scope]:
            name = name[len(assoc_scope) + 1:]
            if name not in exclude_names:
                self.add_association(name=name, assoc_name=name, scope=scope, assoc_scope=assoc_scope)
        for name in self.scope_associations[assoc_scope]:
            name = name[len(assoc_scope) + 1:]
            if name not in exclude_names:
                self.add_association(name=name, assoc_name=name, scope=scope, assoc_scope=assoc_scope)

    def register_variable(self, name, scope = None):
        if scope:
            self._check_scope_conflict(name, scope)
            self._registerd_variables.append("{scope}_{name}".format(scope=scope, name=name))
        else:
            self._check_name_conflict(name)
            self._registerd_variables.append(name)

    def add_variable(self, name, value, scope = None):
        if scope:
            self._check_scope_conflict(name, scope)
            self.variables["{scope}_{name}".format(scope=scope, name=name)] = value
            self.scope_variables[scope].append("{scope}_{name}".format(scope=scope, name=name))
        else:
            self._check_name_conflict(name)
            self.variables[name] = value
            self.name_variables.append(name)
        return self

    def build(self):
        class ArgumentsGetter(object):
            def __init__(self):
                pass
            def __getitem__(self, item):
                return getattr(self, item)
        self.arguments = ArgumentsGetter()
        self.parser.parse_args(namespace=self.arguments)
        self.built = True

    def __getitem__(self, name_or_scope, assoc_find = False):
        if not self.built:
            self.build()
        if name_or_scope in self.scope_arguments or name_or_scope in self.scope_variables or name_or_scope in self.scope_associations:
            args = {}
            for name in self.scope_arguments[name_or_scope]:
                args[name[len(name_or_scope) + 1: ]] = self.arguments[name]
            for name in self.scope_variables[name_or_scope]:
                args[name[len(name_or_scope) + 1: ]] = self.variables[name]
            for name in self.scope_associations[name_or_scope]:
                if isinstance(self.associations[name], list):
                    assoc_scope = self.associations[name][0]
                    assoc_name = self.associations[name][1]
                    value = self.__getitem__(assoc_scope, assoc_find=True)
                    if value and assoc_name in value:
                        value = value[assoc_name]
                    else:
                        value = None
                else:
                    value = self.__getitem__(self.associations[name], assoc_find=True)
                if value:
                    args[name[len(name_or_scope) + 1:]] = value
            for reg_name in self._registerd_variables:
                if reg_name.startswith(name_or_scope) and reg_name not in self.scope_variables[name_or_scope]:
                    if not assoc_find:
                        raise Exception("%s is empty, use add_variable to add the variable" % reg_name)
                    else:
                        return None
            return args
        elif name_or_scope in self.name_arguments:
            return self.arguments[name_or_scope]
        elif name_or_scope in self.variables:
            return self.variables[name_or_scope]
        elif name_or_scope in self.name_associations:
            if isinstance(self.associations[name_or_scope], list):
                assoc_scope = self.associations[name_or_scope][0]
                assoc_name = self.associations[name_or_scope][1]
                value = self.__getitem__(assoc_scope, assoc_find=True)
                if value:
                    value = value[assoc_name]
            else:
                value = self.__getitem__(self.associations[name_or_scope], assoc_find=True)
            if value:
                return value
            else:
                raise Exception("cannot find key: %s" % name_or_scope)
        elif name_or_scope in self._registerd_variables:
            if not assoc_find:
                raise Exception("%s is empty, use add_variable to add the variable" % name_or_scope)
            else:
                return None
        else:
            if not assoc_find:
                raise Exception("cannot find key: %s" % name_or_scope)
            else:
                return None

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