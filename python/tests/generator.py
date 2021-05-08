import os
import pathlib
import pycubool as cb


__all__ = [
    "Generator"
]


def row_matrix_from_vec(v):
    rows = [0] * v.nvals
    cols = v.to_list()
    shape = (1, v.nrows)
    return cb.Matrix.from_lists(shape=shape, rows=rows, cols=cols)


class Generator:

    __slots__ = ["prefix_path"]

    def __init__(self, prefix_path):
        self.prefix_path = prefix_path

    def generate(self):
        # (name of the folder section, function to run, required matrices count (for safety))
        config = [
            ("matrix_dup", self.__matrix_dup, 1),
            ("matrix_ewiseadd", self.__matrix_ewiseadd, 2),
            ("kronecker", self.__kronecker, 2),
            ("mxm", self.__mxm, 2),
            ("mxv", self.__mxv, 2),
            ("reduce", self.__reduce, 1),
            ("reduce_vector", self.__reduce_vector, 1),
            ("transpose", self.__transpose, 1),
            ("vector_dup", self.__vector_dup, 1),
            ("vector_ewiseadd", self.__vector_ewiseadd, 2),
            ("vxm", self.__vxm, 2),
        ]

        # Number of shapes of matrices per test to generate
        shapes = [
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60), (50, 60)], [(500, 600), (500, 600)], [(1000, 2000), (1000, 2000)]],
            [[(10, 20), (50, 60)], [(20, 30), (200, 300)], [(40, 60), (400, 600)]],
            [[(50, 60), (60, 80)], [(500, 600), (600, 700)], [(1000, 2000), (2000, 1500)]],
            [[(50, 60), (1, 60)], [(500, 600), (1, 600)], [(1000, 2000), (1, 2000)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(50, 60)], [(500, 600)], [(1000, 2000)]],
            [[(1, 60)], [(1, 600)], [(1, 2000)]],
            [[(1, 60), (1, 60)], [(1, 600), (1, 600)], [(1, 2000), (1, 2000)]],
            [[(1, 60), (60, 50)], [(1, 600), (600, 500)], [(1, 2000), (2000, 1000)]]
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
            [0.5, 0.05, 0.005],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.05, 0.005]
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
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5],
            [20, 10, 5]
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
                        matrices.append(cb.Matrix.generate(shape=shape, density=density))

                    assert len(matrices) == req_matrices

                    matrices.append(func(matrices))

                    for matrix_idx, matrix in enumerate(matrices):
                        cb.export_matrix_to_mtx(case_path / f"matrix_{matrix_idx}.mtx", matrix)

    @staticmethod
    def __matrix_dup(matrices):
        first_matrix = matrices[0]
        return first_matrix.dup()

    @staticmethod
    def __matrix_ewiseadd(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.ewiseadd(second_matrix)

    @staticmethod
    def __kronecker(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.kronecker(second_matrix)

    @staticmethod
    def __mxm(matrices):
        first_matrix = matrices[0]
        second_matrix = matrices[1]
        return first_matrix.mxm(second_matrix, accumulate=False)

    @staticmethod
    def __mxv(matrices):
        first_matrix = matrices[0]
        second_vector = matrices[1].extract_row(0)
        return row_matrix_from_vec(first_matrix.mxv(second_vector))

    @staticmethod
    def __reduce(matrices):
        first_matrix = matrices[0]
        return first_matrix.reduce()

    @staticmethod
    def __reduce_vector(matrices):
        first_matrix = matrices[0]
        return row_matrix_from_vec(first_matrix.reduce_vector())

    @staticmethod
    def __transpose(matrices):
        first_matrix = matrices[0]
        return first_matrix.transpose()

    @staticmethod
    def __vector_dup(matrices):
        first_vector = matrices[0].extract_row(0)
        return row_matrix_from_vec(first_vector.dup())

    @staticmethod
    def __vector_ewiseadd(matrices):
        first_vector = matrices[0].extract_row(0)
        second_vector = matrices[1].extract_row(0)
        return row_matrix_from_vec(first_vector.ewiseadd(second_vector))

    @staticmethod
    def __vxm(matrices):
        first_vector = matrices[0].extract_row(0)
        second_matrix = matrices[1]
        return row_matrix_from_vec(first_vector.vxm(second_matrix))


if __name__ == "__main__":
    PATH = pathlib.Path(__file__)
    PATH_TO_DATA = PATH.parent.parent / "data"

    g = Generator(PATH_TO_DATA)
    g.generate()
