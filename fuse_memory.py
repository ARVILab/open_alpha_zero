import argparse

from core.utils.utils import serialize, deserialize

import os


def fuse_memory(old_memory_path, new_memory_path, out_memory_path):
    if os.path.isfile(old_memory_path) and os.path.isfile(new_memory_path):
        try:
            serialize(deserialize(new_memory_path) + deserialize(old_memory_path), out_memory_path)
        except:
            print("Could not deserialize new + old. Try reverse order")
            serialize(deserialize(old_memory_path) + deserialize(new_memory_path), out_memory_path)
    elif os.path.isfile(new_memory_path):
        serialize(deserialize(new_memory_path), out_memory_path)


def clean_dir(dir_path):
    files = glob.glob(dir_path + '/*')
    for f in files:
        os.remove(f)


def throw_error(message):
    print(message)
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--first_memory_path", dest="first_memory_path",
                        help="")

    parser.add_argument("--second_memory_path", dest="second_memory_path",
                        help="")

    parser.add_argument("--out_memory_path", dest="out_memory_path",
                        help="")

    options = parser.parse_args()

    if not options.first_memory_path:
        parser.error('first_memory_path must be selected')

    if not options.second_memory_path:
        parser.error('second_memory_path must be selected')

    if not options.out_memory_path:
        print("fuse into ", options.first_memory_path)
        fuse_memory(options.first_memory_path, options.second_memory_path, options.first_memory_path)
    else:
        print("fuse into ", options.out_memory_path)
        fuse_memory(options.first_memory_path, options.second_memory_path, options.out_memory_path)

    print("fuse finished!")
