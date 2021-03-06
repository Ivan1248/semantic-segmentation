from ioutils import path


def find_in_ancestor(path_end):
    try:
        return path.find_in_ancestor(__file__, path_end)
    except:
        print("ERROR: dirs.py: Could not find '"+path_end+"'.")
        return None


SAVED_MODELS = find_in_ancestor('storage/models')
LOGS = find_in_ancestor('storage/logs')
DATASETS = find_in_ancestor('storage/datasets')