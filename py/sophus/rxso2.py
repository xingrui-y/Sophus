import sympy
import sys
import unittest
import sophus
import functools


class RxSo2:
    """ 2 dimensional group of scaled-orthogonal matrices with positive
        determinant """

    def __init__(self, z):
        """ internally represented by a complex number z """
        self.z = z

    def scale(self):
        return sympy.sqrt(self.z.squared_norm())

    @staticmethod
    def exp(theta_sigma):
        """ exponential map """
        scale = sympy.exp(theta_sigma[1])
        z = sophus.So2.exp(theta_sigma[0]).z
        return RxSo2(sophus.Complex(scale * z.real, scale * z.imag))

    def log(self):
        """ logarithmic map"""

        sigma = sympy.log(self.scale())
        theta = sophus.So2(self.z).log()
        return sophus.Vector2(theta, sigma)

    def __repr__(self):
        return "RxSo2:" + repr(self.z)

    @staticmethod
    def hat(theta_sigma):
        return sophus.Matrix([[theta_sigma[1], -theta_sigma[0]],
                              [theta_sigma[0], theta_sigma[1]]])

    def matrix(self):
        """ returns matrix representation """
        s = self.scale()
        return sophus.Matrix([[s * self.z.real, -s * self.z.imag],
                              [s * self.z.imag, s * self.z.real]])

    def __mul__(self, right):
        """ left-multiplication
            either rotation concatenation or point-transform """
        if isinstance(right, sophus.Vector2):
            return self.matrix() * right
        elif isinstance(right, So2):
            return RxSo2(self.z * right.z)
        assert False, "unsupported type: {0}".format(type(right))

    def __getitem__(self, key):
        return self.z[key]

    @staticmethod
    def calc_Dx_exp_x(x):
        return sophus.Matrix(2, 2,
                             lambda r, c: sympy.diff(RxSo2.exp(x)[c], x[r]))

    @staticmethod
    def Dx_exp_x_at_0():
        return sophus.Matrix([[0, 1], [1, 0]])

    @staticmethod
    def calc_Dx_exp_x_at_0(x):
        return RxSo2.calc_Dx_exp_x(x).subs(x[0], 0).limit(x[1], 0)

    @staticmethod
    def Dxi_x_matrix(x, i):
        n = sympy.sqrt(x.z.squared_norm())
        x00 = x[0]**2
        x01 = x[0] * x[1]
        x11 = x[1]**2
        if i == 0:
            return sophus.Matrix([[x00 / n + n, -x01 / n],
                                  [x01 / n, x00 / n + n]])
        if i == 1:
            return sophus.Matrix([[x01 / n, -x11 / n - n],
                                  [x11 / n + n, x01 / n]])
        if i == 2:
            return sophus.Matrix([[x01 / n, -x11 / n - n],
                                  [x11 / n + n, x01 / n]])

    @staticmethod
    def calc_Dxi_x_matrix(x, i):
        return sophus.Matrix(2, 2,
                             lambda r, c: sympy.diff(x.matrix()[r, c], x[i]))

    @staticmethod
    def Dxi_exp_x_matrix(x, i):
        sR = RxSo2.exp(x)
        Dx_exp_x = RxSo2.calc_Dx_exp_x(x)
        l = [RxSo2.Dxi_x_matrix(sR, j) * Dx_exp_x[i, j] for j in [0, 1]]
        return functools.reduce((lambda a, b: a + b), l)

    @staticmethod
    def calc_Dxi_exp_x_matrix(x, i):
        return sophus.Matrix(
            2, 2, lambda r, c: sympy.diff(RxSo2.exp(x).matrix()[r, c], x[i]))

    @staticmethod
    def Dxi_exp_x_matrix_at_0(i):
        v = sophus.Vector2.zero()
        v[i] = 1
        return RxSo2.hat(v)

    @staticmethod
    def calc_Dxi_exp_x_matrix_at_0(x, i):
        return sophus.Matrix(
            2, 2,
            lambda r, c: sympy.diff(RxSo2.exp(x).matrix()[r, c], x[i])).subs(
                x[0], 0).limit(x[1], 0)


class TestRxSo2(unittest.TestCase):
    def setUp(self):
        sigma, theta = sympy.symbols('sigma theta', real=True)
        self.theta_sigma = sophus.Vector2(theta, sigma)
        x, y = sympy.symbols('x y', real=True)
        p0, p1 = sympy.symbols('p0 p1', real=True)
        self.a = RxSo2(sophus.Complex(x, y))
        self.p = sophus.Vector2(p0, p1)

    def test_exp_log(self):
        for theta_sigma in [
                sophus.Vector2(0.01, 0.5),
                sophus.Vector2(0.2, -0.25)
        ]:
            w = RxSo2.exp(theta_sigma).log()
            for i in range(0, 2):
                self.assertAlmostEqual(theta_sigma[i], w[i])

    def test_matrix(self):
        sR_foo_bar = RxSo2.exp(self.theta_sigma)
        sRmat_foo_bar = sR_foo_bar.matrix()
        point_bar = self.p
        p1_foo = sR_foo_bar * point_bar
        p2_foo = sRmat_foo_bar * point_bar
        self.assertEqual(
            sympy.simplify(p1_foo - p2_foo), sophus.Vector2.zero())

    def test_derivatives(self):
        self.assertEqual(
            sympy.simplify(
                RxSo2.calc_Dx_exp_x_at_0(self.theta_sigma) -
                RxSo2.Dx_exp_x_at_0()), sophus.Matrix.zeros(2, 2))

        for i in [0, 1, 2]:
            self.assertEqual(
                sympy.simplify(
                    RxSo2.calc_Dxi_x_matrix(self.a, i) -
                    RxSo2.Dxi_x_matrix(self.a, i)), sophus.Matrix.zeros(2, 2))

        for i in [0, 1]:
            self.assertEqual(
                sympy.simplify(
                    RxSo2.Dxi_exp_x_matrix(self.theta_sigma, i) -
                    RxSo2.calc_Dxi_exp_x_matrix(self.theta_sigma, i)),
                sophus.Matrix.zeros(2, 2))

            print(RxSo2.calc_Dxi_exp_x_matrix_at_0(self.theta_sigma, i))
            #print(RxSo2.Dxi_exp_x_matrix_at_0(i))

            self.assertEqual(
                sympy.simplify(
                    RxSo2.Dxi_exp_x_matrix_at_0(i) -
                    RxSo2.calc_Dxi_exp_x_matrix_at_0(self.theta_sigma, i)),
                sophus.Matrix.zeros(2, 2))

    # def test_codegen(self):
    #     stream = sophus.cse_codegen(RxSo2.calc_Dx_exp_x(self.theta_sigma))
    #     filename = "cpp_gencode/RxSo2_Dx_exp_x.cpp"

    #     # set to true to generate codegen files
    #     if False:
    #         file = open(filename, "w")
    #         for line in stream:
    #             file.write(line)
    #         file.close()
    #     else:
    #         file = open(filename, "r")
    #         file_lines = file.readlines()
    #         for i, line in enumerate(stream):
    #             self.assertEqual(line, file_lines[i])
    #         file.close()
    #     stream.close


if __name__ == '__main__':
    unittest.main()
