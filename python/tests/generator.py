import os
import pathlib
import pycubool
import pycubool.io as io


__all__ = [
    "Generator"
]


class Generator:

    __slots__ = ["prefix_path"]

    def __init__(self, prefix_path, ):
        self.prefix_path = prefix_path

    def generate(self):
        # (name of the folder section, function to run, required matrices count (for safety))
        config = [
            ("dup", self._dup, 1),
            ("ewiseadd", self._ewiseadd, 2),
            ("kronecker", self._kronecker, 2),
            ("mxm", self._mxm, 2),
            ("transpose", self._transpose, 1),
            ("reduce", self._reduce, 1),
            ("to_lists", self._to_lists, 1)
        ]

        # Number of shapes of matrices per test to generate
        shapes = [
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60), (50, 60)], [(500, 600), (500, 600)], [(1000, 2000), (1000, 2000)]],
            [[(10, 20), (50, 60)], [(20, 30), (200, 300)], [(40, 60), (400, 600)]],
            [[(50, 60), (60, 80)], [(500, 600), (600, 700)], [(1000, 2000), (2000, 1500)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
        ]

        # Densities per shape
        densities = [
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
            [0.5, 0.05, 0.005],
        ]

        # Number of test runs per each shape family
        cases = [
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
        ]

        total = len(config)

        assert total == len(shapes)
        assert total == len(densities)
        assert total == len(cases)

        for cfg, f_shapes, f_densities, f_cases in zip(config, shapes, densities, cases):
            name, func, req_matrices = cfg

            test_path = self.prefix_path / f"func_{name}"
            os.mkdir(test_path)

            for shapes_idx, g_shapes in enumerate(f_shapes):

                shape_path = test_path / f"shape_{shapes_idx}"
                os.mkdir(shape_path)

                density = f_densities[shapes_idx]
                cases_to_gen = f_cases[shapes_idx]

                for case_idx in range(cases_to_gen):

                    case_path = shape_path / f"case_{case_idx}"
                    os.mkdir(case_path)

                    matrices = list()
                    for shape in g_shapes:
                        matrices.append(pycubool.Matrix.generate(shape=shape, density=density))

                    assert len(matrices) == req_matrices

                    matrices.append(func(matrices))

                    for matrix_idx, matrix in enumerate(matrices):
                        pycubool.io.export_matrix_to_mtx(case_path / f"matrix_{matrix_idx}.mtx", matrix)

    @staticmethod
    def _dup(matrices):
        first_matrix = matrices[0]
        return first_matrix.dup()

    @staticmethod
    def _ewiseadd(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.ewiseadd(second_matrix)

    @staticmethod
    def _kronecker(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.kronecker(second_matrix)

    @staticmethod
    def _mxm(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.mxm(second_matrix, accumulate=False)

    @staticmethod
    def _transpose(matrices):
        first_matrix = matrices[0]
        return first_matrix.transpose()

    @staticmethod
    def _reduce(matrices):
        first_matrix = matrices[0]
        return first_matrix.reduce()

    @staticmethod
    def _to_lists(matrices):
        first_matrix = matrices[0]
        return first_matrix


if __name__ == "__main__":
    PATH = pathlib.Path(__file__)
    PATH_TO_DATA = PATH.parent.parent / "data"

    g = Generator(PATH_TO_DATA)
    g.generate()
