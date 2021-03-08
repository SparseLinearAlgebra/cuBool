import os
import pathlib


__all__ = [
    "Config",
    "cfg"
]


class Config:

    __slots__ = ["prefix_path"]

    def __init__(self, prefix_path):
        self.prefix_path = prefix_path

    def get_test_cases(self, folder, req_args):
        folder = "func_" + folder

        path = self.prefix_path / folder
        total = 0
        matrices = [[] for _ in range(req_args)]

        for shape in os.listdir(path):
            shape_path = path / shape
            for case in os.listdir(shape_path):
                case_path = shape_path / case
                total += 1

                files = list(os.listdir(case_path))
                files.sort()

                for i, file in enumerate(files):
                    file_path = case_path / file
                    matrices[i].append(file_path)

        return matrices, total


PATH = pathlib.Path(__file__)
PATH_TO_DATA = PATH.parent.parent / "data"

# Use this variable to pass global config settings into tests
cfg = Config(PATH_TO_DATA)
