class Params(object):
    """ A simple dictionary that has its keys as attributes available. """

    def __init__(self, keys=None):
        if keys is not None:
            for k, v in keys.items():
                self.__dict__[k] = v

    def __str__(self):
        s = ""
        for name in sorted(self.__dict__.keys()):
            s += "%-18s %s\n" % (name + ":", self.__dict__[name])
        return s

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.__dict__

    def update_values(self, new_params):
        """
        Merge config (keys, values)  into this Params object.  For example, `cfg_list = ['FOO.BAR', 0.5]`.

        Parameters:
            new_params: either a dict or list of even number of elements. If a dict, updates each config
                        in ``self`` with values from ``new_params``.
                        If a list, it is interpreted as [key1, value1, key2, value2,...]. Useful for
                        passing in data directly from the command line.
        """
        if isinstance(new_params, list):
            assert len(new_params) % 2 == 0, "Unexpected length: " + str(len(new_params))
            new_params = {k: parse_val(v) for (k, v) in zip(new_params[0::2], new_params[1::2])}

        for key, value in new_params.items():
            self.__dict__[key] = value


def parse_val(val):
    """
    Parses a string into int/float/bool or list of strings.

    :param val:
    :return:
    """

    try:
        val, found = int(val), True
    except ValueError:
        found = False
    # Try float
    if not found:
        try:
            val, found = float(val), True
        except ValueError:
            found = False
    # Try boolean
    if not found:
        if val == 'False':
            val, found = False, True
        elif val == 'True':
            val, found = True, True
        else:
            found = False
    # Try list
    if not found:
        if val[0] == '[' and val[-1] == ']':
            val = val[1:-1].split(',')
            val = [x.strip() for x in val]
        else:
            found = False
    return val
