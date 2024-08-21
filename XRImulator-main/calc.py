# print((8.5 * 3 + 8.5 * 6 + (7.5 *3 + 8.5) * 6 + 7 * 3 + 12 * 8.5 + 7.5 * 6 + 8 * 6 + 7.5 * 6) / 66)
import numpy as np
import scipy.constants as spc

# test1 = np.array([1,2,3])
# test2 = np.array([1,2,4])

# test = lambda x, y, z: z * x * y

# print(np.sum(test(5,test1, test2)))

# F = lambda D, theta: D/(2*np.tan(theta/2))
# theta = lambda D, F: 2 * np.arctan(D / (2*F))
# Ds = np.array([.035, .105, .315, .945])
# Fs = np.array([1.2, 3.7, 11.1, 33.4]) * 1e3
# print(theta(Ds, Fs) * 3600 * 360 / (2*np.pi))
# print(np.log10(-(0.0004371 - 1.616e-7) / (0.02 - 10)))

# test = dict(D = 5, F = 7)

# test_array = np.array([test, test, test])
# test['D'] += 5
# print(test['D'])

# test = np.array([True, True, False])
# print(test.nonzero()[0].size)

print(12*1e9 / (3.2 * 1e3 * spc.parsec) * 360 * 3600 / (2*np.pi))
print(16.77 * 696*1e8 * 1e-11)
