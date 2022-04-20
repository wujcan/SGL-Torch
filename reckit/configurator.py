__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Configurator"]

import os
import sys
from configparser import ConfigParser
from collections import OrderedDict


class Configurator(object):
    """A configurator class.

    This class can read arguments from ini-style configuration file and parse
    arguments from command line simultaneously. This class can also convert
    the argument value from `str` to `int`, `float`, `bool`, `list` and `None`
    automatically. The arguments from command line have the highest priority
    than that from configuration file. That is, if there are same argument
    name in configuration file and command line, the value in the former will
    be overwritten by that in the latter, whenever the command line is phased
    before or after read ini files. Moreover:

    * Command line: The format of arguments is ``--arg_name=arg_value``,
      For example::

        python main.py --model=Pop --num_thread=8 --metric=["Recall", "NDCG"]

    * Configuration file: This file must be ini-style. If there is only one
      section and whatever the name is, this class will read arguments from
      that section. If there are more than one sections, this class will read
      arguments from the section named `section`.

    After phasing or reading arguments successfully, arguments can be accessed
    by index or as property:

        config = Configurator()
        config.parse_cmd()
        num_thread = config["num_thread"]
        metric = config.metric

    Here, the arguments of `num_thread` and `metric` are automatically convert to
    `int` and `list`, respectively.
    """

    def __init__(self, root_dir, data_dir):
        """Initializes a new `Configurator` instance.
        """
        self.root_dir = root_dir
        self.data_dir = data_dir
        self._sections = OrderedDict()
        self._cmd_args = OrderedDict()
        self._summary_id = None

    def add_config(self, cfg_file, section="default", used_as_summary=False):
        """Read and add config from ini-style file.

        Args:
            cfg_file (str): The path of ini-style configuration file.
            section (str): The section of configuration to be read. 'section'
                will be activated only if there are more than one sections
                in configuration file, i.e. if there is only one section and
                whatever the name is, the arguments will be read from it.
            used_as_summary (bool): Whether used to get the summary string.

        Raises:
            FileNotFoundError: If cfg_file does not exist.
            ValueError: If 'cfg_file' is empty, or
                if 'cfg_file' has more than one sections but no 'section'.
        """
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError("File '%s' does not exist." % cfg_file)

        config = ConfigParser()
        config.optionxform = str
        config.read(cfg_file, encoding="utf-8")
        sections = config.sections()

        if len(sections) == 0:
            raise ValueError("'%s' is empty!" % cfg_file)
        elif len(sections) == 1:
            config_sec = sections[0]
        elif section in sections:
            config_sec = section
        else:
            raise ValueError("'%s' has more than one sections but there is no "
                             "section named '%s'" % (cfg_file, section))

        # Generate the section name
        sec_name = "%s:[%s]" % (os.path.basename(cfg_file).split(".")[0], config_sec)
        if sec_name in self._sections:
            sec_name += "_%d" % len(self._sections)

        if used_as_summary is True:
            # record the section name if this section is used to
            # get the summary of this configurator
            self._summary_id = sec_name

        config_arg = OrderedDict(config[config_sec].items())
        # update arguments from the command line
        for arg in self._cmd_args:
            if arg in config_arg:
                config_arg[arg] = self._cmd_args[arg]

        self._sections[sec_name] = config_arg

    def parse_cmd(self):
        """Parse the arguments from command line.

        Notes:
            The arguments from command line will cover the arguments read from ini
            files, whenever this method is called before or after calling 'add_config'.

            The format of command line:
                python main.py --model Pop --num_thread 128 --group_view [10,30,50,100]

        Raises:
            SyntaxError: If the arguments in command have invalid formats.
        """
        args = sys.argv[1:]
        # if len(args) % 2 != 0:
        #     raise SyntaxError("The numbers of arguments and its values are not equal.")
        # for arg_name, arg_value in zip(args[0::2], args[1::2]):
        #     if not arg_name.startswith("--"):
        #         raise SyntaxError("Command arg must start with '--', but '%s' is not!" % arg_name)
        #     self._cmd_args[arg_name[2:]] = arg_value
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--"):
                    raise SyntaxError("Commend arg must start with '--', but '%s' is not!" % arg)
                arg_name, arg_value = arg[2:].split("=")
                self._cmd_args[arg_name] = arg_value

        # cover the arguments from ini files
        for sec_name, sec_arg in self._sections.items():
            for cmd_argn, cmd_argv in self._cmd_args.items():
                if cmd_argn in sec_arg:
                    sec_arg[cmd_argn] = cmd_argv

    def summarize(self):
        """Get a summary of the configurator's arguments.

        Returns:
            str: A string summary of arguments.

        Raises:
            ValueError: If configurator is empty.
        """
        if len(self._sections) == 0:
            raise ValueError("Configurator is empty.")
        if self._summary_id is not None:
            args = self._sections[self._summary_id]
        else:
            args = self._sections[next(reversed(self._sections.keys()))]

        params_id = '_'.join(["{}={}".format(arg, value) for arg, value in args.items() if len(value) < 20])
        special_char = {'/', '\\', '\"', ':', '*', '?', '<', '>', '|', '\t', '\n', '\r', '\v', ' '}
        params_id = [c if c not in special_char else '_' for c in params_id]
        params_id = ''.join(params_id)
        return params_id

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("Index must be a str.")
        for sec_name, sec_args in self._sections.items():
            if item in sec_args:
                param = sec_args[item]
                break
        else:
            if item in self._cmd_args:
                param = self._cmd_args[item]
            else:
                raise KeyError("There are not the argument named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except (NameError, SyntaxError):
            if param.lower() == "true":
                value = True
            elif param.lower() == "false":
                value = False
            else:
                value = param

        return value

    def __getattr__(self, item):
        return self[item]

    def __contains__(self, o):
        for sec_name, sec_args in self._sections.items():
            if o in sec_args:
                flag = True
                break
        else:
            if o in self._cmd_args:
                flag = True
            else:
                flag = False

        return flag

    def __str__(self):
        sec_str = []

        # ini files
        for sec_name, sec_args in self._sections.items():
            arg_info = '\n'.join(["{}={}".format(arg, value) for arg, value in sec_args.items()])
            arg_info = "%s:\n%s" % (sec_name, arg_info)
            sec_str.append(arg_info)

        # cmd
        if self._cmd_args:
            cmd_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self._cmd_args.items()])
            cmd_info = "Command line:\n%s" % cmd_info
            sec_str.append(cmd_info)

        info = '\n\n'.join(sec_str)
        return info

    def __repr__(self):
        return self.__str__()
