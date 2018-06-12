import pickle

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


# ----------------------------------------------------------------------
def serialize(object, path):
    """
    Pickle a Python object
    """
    with open(path, "wb") as pfile:
        pickle.dump(object, pfile)


# ----------------------------------------------------------------------
def deserialize(path):
    """
    Extracts a pickled Python object and returns it
    """
    with open(path, "rb") as pfile:
        object = pickle.load(pfile)
    return object
