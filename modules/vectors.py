from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

import numpy as np


class Vector(object):

    def __init__(self, coordinates: iter, tolerance: float = 1e-10) -> None:
        try:
            if not coordinates:
                raise ValueError

            self.coordinates = np.array(coordinates, dtype=np.float64)
            self.dimension = len(coordinates)
            self.tolerance = tolerance

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')

    def __str__(self) -> str:
        return f'Vector: {self.coordinates}'

    def __eq__(self, other: Vector) -> bool:
        assert isinstance(other, Vector)

        return self.coordinates == other.coordinates

    def __add__(self, other: Vector) -> Vector:
        assert isinstance(other, Vector)
        assert self.dimension == other.dimension

        coord = [i + j for i, j in zip(self.coordinates, other.coordinates)]

        return Vector(coordinates=coord)

    def __sub__(self, other: Vector) -> Vector:
        assert isinstance(other, Vector)
        assert self.dimension == other.dimension

        coord = [i - j for i, j in zip(self.coordinates, other.coordinates)]

        return Vector(coordinates=coord)

    def scalar_mul(self, scalar: (int, float)) -> Vector:
        assert isinstance(scalar, (int, float))

        coordinates = [coord * scalar for coord in self.coordinates]

        return Vector(coordinates=coordinates)

    @property
    def magnitude(self) -> float:
        return np.sqrt(np.sum(np.power(self.coordinates, 2)))

    @property
    def direction(self) -> Vector:
        try:
            unit_coord = [coord / self.magnitude for coord in self.coordinates]
            return Vector(coordinates=unit_coord)

        except ZeroDivisionError:
            raise Exception('Cannot normalize the zero vector')

    def dot_product(self, other):
        assert isinstance(other, Vector)

        return sum(
            [i * j for i, j in zip(self.coordinates, other.coordinates)]
        )

    def get_angle_with(self, other: Vector, in_degrees: bool = False) -> float:
        assert isinstance(other, Vector)

        unit_dot_prod = self.direction.dot_product(other.direction)
        angle = np.arccos(unit_dot_prod)

        if in_degrees:
            angle *= 180/np.pi

        return angle

    @property
    def is_null_vector(self) -> bool:
        return self.magnitude < self.tolerance

    def is_parallel_to(self, other: Vector) -> bool:
        assert isinstance(other, Vector)

        return (
            self.is_null_vector or
            other.is_null_vector or
            self.get_angle_with(other=other) in (0, np.pi)
        )

    def is_orthogonal_to(self, other: Vector) -> bool:
        assert isinstance(other, Vector)

        return np.abs(self.dot_product(other=other)) < self.tolerance

    def component_parallel_to(self, basis):
        assert isinstance(basis, Vector)

        if self.is_null_vector or basis.is_null_vector:
            raise Exception('Cannot project from/to null vector')

        u = basis.direction
        weight = self.dot_product(other=u)

        return u.scalar_mul(scalar=weight)

    def component_orthogonal_to(self, basis: Vector) -> Vector:
        assert isinstance(basis, Vector)

        if self.is_null_vector or basis.is_null_vector:
            raise Exception('Cannot project from/to null vector')

        projection = self.component_parallel_to(basis=basis)
        rejection = self - projection

        return rejection

    @staticmethod
    def pad_third_dim(vector: Vector) -> Vector:
        if vector.dimension == 2:
            new_coordinates = vector.coordinates + (0, )
            vector = Vector(coordinates=new_coordinates)

        if vector.dimension != 3:
            raise Exception(f'Wrong number of dimensions for vector {vector}')

        return vector

    def cross_product(self, other: Vector) -> Vector:
        assert isinstance(other, Vector)

        x1, y1, z1 = self.pad_third_dim(vector=self).coordinates
        x2, y2, z2 = self.pad_third_dim(vector=other).coordinates

        return Vector([
            y1*z2 - y2*z1,
            -(x1*z2 - x2*z1),
            x1*y2 - x2*y1
        ])

    def get_area_parallelogram(self, other: Vector) -> float:
        assert isinstance(other, Vector)

        return self.cross_product(other=other).magnitude

    def get_area_triangle(self, other: Vector) -> float:
        assert isinstance(other, Vector)

        return self.get_area_parallelogram(other=other) / 2


if __name__ == '__main__':
    # Test case 1
    vector1 = Vector(coordinates=(8.218, -9.341))
    vector2 = Vector(coordinates=(-1.129, 2.111))
    print(vector1 + vector2)

    # Test case 2
    vector3 = Vector(coordinates=(7.119, 8.215))
    vector4 = Vector(coordinates=(-8.223, 0.878))
    print(vector3 - vector4)

    # Test case 3
    vector5 = Vector(coordinates=(1.671, -1.012, -0.318))
    print(vector5.scalar_mul(scalar=7.41))

    # Test case 4
    vector6 = Vector(coordinates=(-0.221, 7.437))
    print(vector6.magnitude)

    vector7 = Vector(coordinates=(8.813, -1.331, 6.247))
    print(vector7.magnitude)

    # Test case 5
    vector8 = Vector(coordinates=(5.581, -2.136))
    print(vector8.direction)

    vector9 = Vector(coordinates=(1.996, 3.108, -4.554))
    print(vector9.direction)

    # Test case 6
    vector10 = Vector(coordinates=(7.887, 4.138))
    vector11 = Vector(coordinates=(-8.802, 6.776))
    print(vector10.dot_product(other=vector11))

    vector12 = Vector(coordinates=(-5.955, -4.904, -1.874))
    vector13 = Vector(coordinates=(-4.496, -8.755, 7.103))
    print(vector12.dot_product(other=vector13))

    # Test case 7
    vector14 = Vector(coordinates=(3.183, -7.627))
    vector15 = Vector(coordinates=(-2.668, 5.319))
    print(vector14.get_angle_with(other=vector15))

    vector16 = Vector(coordinates=(7.35, 0.221, 5.188))
    vector17 = Vector(coordinates=(2.751, 8.259, 3.985))
    print(vector16.get_angle_with(other=vector17, in_degrees=True))

    # Test case 8
    vector18 = Vector(coordinates=(-7.579, -7.88))
    vector19 = Vector(coordinates=(22.737, 23.64))
    print('Are parallel?', vector18.is_parallel_to(other=vector19))
    print('Are orthogonal?', vector18.is_orthogonal_to(other=vector19))

    vector20 = Vector(coordinates=(-2.029, 9.97, 4.172))
    vector21 = Vector(coordinates=(-9.231, -6.639, -7.245))
    print('Are parallel?', vector20.is_parallel_to(other=vector21))
    print('Are orthogonal?', vector20.is_orthogonal_to(other=vector21))

    vector22 = Vector(coordinates=(-2.328, -7.284, -1.214))
    vector23 = Vector(coordinates=(-1.821, 1.072, -2.94))
    print('Are parallel?', vector22.is_parallel_to(other=vector23))
    print('Are orthogonal?', vector22.is_orthogonal_to(other=vector23))

    vector24 = Vector(coordinates=(2.118, 4.827))
    vector25 = Vector(coordinates=(0, 0))
    print('Are parallel?', vector24.is_parallel_to(other=vector25))
    print('Are orthogonal?', vector24.is_orthogonal_to(other=vector25))

    # Test case 9
    vector26 = Vector(coordinates=(3.039, 1.879))
    vector27 = Vector(coordinates=(0.825, 2.036))
    print(vector26.component_parallel_to(basis=vector27))

    # Test case 10
    vector28 = Vector(coordinates=(-9.88, -3.264, -8.159))
    vector29 = Vector(coordinates=(-2.155, -9.353, -9.473))
    print(vector28.component_orthogonal_to(basis=vector29))

    # Test case 11
    vector30 = Vector(coordinates=(3.009, -6.172, 3.692, -2.51))
    vector31 = Vector(coordinates=(6.404, -9.144, 2.759, 8.718))
    v_parallel = vector30.component_parallel_to(basis=vector31)
    v_orthogonal = vector30.component_orthogonal_to(basis=vector31)
    v = v_parallel + v_orthogonal
    print(v_parallel, v_orthogonal)

    # Test case 12
    vector32 = Vector(coordinates=(8.462, 7.893, -8.187))
    vector33 = Vector(coordinates=(6.984, -5.975, 4.778))
    print(vector32.cross_product(other=vector33))

    # Test case 13
    vector34 = Vector(coordinates=(-8.987, -9.838, 5.031))
    vector35 = Vector(coordinates=(-4.268, -1.861, -8.866))
    print(vector34.get_area_parallelogram(other=vector35))

    # Test case 14
    vector36 = Vector(coordinates=(1.5, 9.547, 3.691))
    vector37 = Vector(coordinates=(-6.007, 0.124, 5.772))
    print(vector36.get_area_triangle(other=vector37))
