import sys
import numpy as np
import argparse
import array
import matplotlib.pyplot as plt
from PIL import Image

N = 512
M = 512

SLIT_HEIGHT = 10
SLIT_WIDTH = 100

PI = np.pi
ENERGY = 30
LIMBDA = 1e-6 * 1239.8 / (ENERGY * 1e3)
K = 2 * PI / LIMBDA
MU = 3.0235443337462318
BETA = MU * LIMBDA / (4 * PI)
DELTA = 5.43e-4 / (ENERGY ** 2)
FOCAL = 8000
THETA = 130


def find_max(data):

    max_value = np.abs(data[0][0])
    for row in data:
        for number in row:
            if np.abs(number) > np.abs(max_value):
                max_value = number

    return max_value


def spectrum_save_png(filename, N, M, data):

    image = Image.fromarray(data.reshape(N, M), mode='L')
    image.save(filename)


def func_sqrt(x, y, z):

  return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def func_optic_t(y):

  return y * y / (2 * FOCAL * DELTA)

parser = argparse.ArgumentParser(description = "Options")
parser.add_argument("value", type = int, help = "Number of option")
args = parser.parse_args()

in_data = np.zeros((N, M), dtype = np.complex128)
four_data = np.zeros((N, M), dtype = np.complex128)

for i in range(N):
    for j in range(M):
        fx = i - N / 2
        fy = j - M / 2

        if abs(fx) < SLIT_HEIGHT and abs(fy) < SLIT_WIDTH:
            if args.value == 1:
                one = ENERGY * np.exp(complex(0, K * func_sqrt(fx, fy, FOCAL))) / func_sqrt(fx, fy, FOCAL)
                two_1 = complex(0, -K)
                two_2 = complex(DELTA, -BETA)
                two = np.exp(two_1 * two_2 * func_optic_t(fy))
                three = np.exp(complex(0, PI / LIMBDA * (fx ** 2 + fy ** 2)))

                in_data[i][j] = one * two * three

                four = np.exp(complex(0, 2 * PI / LIMBDA * (FOCAL + (fx ** 2 + fy ** 2) / (2 * FOCAL))))
                four_data[i][j] = four * complex(0, -1 / LIMBDA)
            elif args.value == 2:
                in_data[i][j] = 2 * 10 * np.cos(PI * SLIT_HEIGHT * np.sin(THETA) / LIMBDA)
            elif args.value == 3:
                in_data[i][j] = np.sinc(PI * SLIT_HEIGHT * fx / LIMBDA / FOCAL) * np.sinc(PI * SLIT_WIDTH * fy / LIMBDA / FOCAL)
            else:
                in_data[i][j] = 1.0
        else:
            in_data[i][j] = 0.0

        if fy > - 60 and fy < 60:
            in_data[i][j] = 0.0


max_in = np.abs(find_max(in_data))
if max_in == 0:
    print("Max value = 0")
    sys.exit()

print(max_in)

spectrum_in = np.abs(in_data) * 255 / max_in
phase_in = np.angle(in_data) * 255 / PI


spectrum_save_png("inputSpectrum.png", N, M, spectrum_in.astype(np.uint8))
spectrum_save_png("inputPhase.png", N, M, phase_in.astype(np.uint8))


out_data = np.fft.fft2(in_data)
out2_data = np.fft.fftshift(out_data)

a = 1
if args.value == 1:
    a = four_data

result = out2_data * a
max_out = np.abs(find_max(result))
if max_out == 0:
    print("Max value = 0")
    sys.exit()

print(max_out)


spectrum_out = np.abs(result) * 255 / max_out
phase_out = np.angle(result) * 255 / PI

spectrum_save_png("outputSpectrum.png", N, M, spectrum_out.astype(np.uint8))
spectrum_save_png("outputPhase.png", N, M, phase_out.astype(np.uint8))


ylist1 = np.sum(spectrum_out, axis = 1)
xlist1 = np.arange(len(ylist1))

ylist2 = np.sum(spectrum_out, axis = 0)
xlist2 = np.arange(len(ylist2))


plt.subplot(2, 2, 1)
plt.plot(xlist1, ylist1)
plt.subplot(2, 2, 2)
plt.plot(ylist2, xlist2)

plt.title('Projection of the result')

plt.show()

if args.value == 10:
    xmin = -0.1
    xmax = 0.1

    count = 100

    xlist = np.linspace(xmin, xmax, count)
    ylist = [func_optic_t(x) for x in xlist]

    plt.plot(xlist, ylist)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph')

    plt.show()

