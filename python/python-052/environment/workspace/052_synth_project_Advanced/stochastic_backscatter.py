
import numpy as np

class CMRGStream:
    M1 = 2147483563
    M2 = 2147483399
    A1 = 40014
    A2 = 40692

    def __init__(self, seed1=12345, seed2=67890):
        self.s1 = int(seed1) % self.M1
        self.s2 = int(seed2) % self.M2
        if self.s1 == 0:
            self.s1 = 1
        if self.s2 == 0:
            self.s2 = 1

    def _advance(self):
        self.s1 = (self.A1 * self.s1) % self.M1
        self.s2 = (self.A2 * self.s2) % self.M2
        z = self.s1 - self.s2
        if z <= 0:
            z += self.M1 - 1
        return z

    def uniform_01(self):
        z = self._advance()
        return z / self.M1

    def normal_pair(self):
        u1 = self.uniform_01()
        u2 = self.uniform_01()
        if u1 <= 0:
            u1 = 1e-10
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        return r * np.cos(theta), r * np.sin(theta)


class StochasticBackscatter:

    def __init__(self, Nx, Ny, Lx, Ly, epsilon=1e-9, k_c=None, seed=42):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.epsilon = epsilon
        if k_c is None:
            self.k_c = (2.0 * np.pi / Lx) * (Nx / 3.0)
        else:
            self.k_c = k_c


        self.rng_real = CMRGStream(seed, seed + 1)
        self.rng_imag = CMRGStream(seed + 2, seed + 3)


        from numpy.fft import fftfreq
        self.kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.ksq = self.KX**2 + self.KY**2

    def _cutoff(self, k):
        kappa = k / self.k_c
        return 0.5 * (1.0 + np.tanh((1.0 - kappa) / 0.1))

    def generate_forcing(self, dt):
        k = np.sqrt(self.ksq)
        k = np.maximum(k, 1e-12)

        C_back = (self.epsilon ** (2.0 / 3.0)) * (k ** (-5.0 / 3.0)) * self._cutoff(k)

        C_back[k > 1.5 * self.k_c] = 0.0


        noise_real = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        noise_imag = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        for i in range(self.Nx):
            for j in range(self.Ny):
                n1, n2 = self.rng_real.normal_pair()
                noise_real[i, j] = n1
                noise_imag[i, j] = n2


        noise_complex = (noise_real + 1j * noise_imag) / np.sqrt(2.0)

        noise_complex[0, 0] = np.real(noise_complex[0, 0])
        if self.Nx % 2 == 0:
            noise_complex[self.Nx // 2, :] = np.real(noise_complex[self.Nx // 2, :])
        if self.Ny % 2 == 0:
            noise_complex[:, self.Ny // 2] = np.real(noise_complex[:, self.Ny // 2])

        amplitude = np.sqrt(2.0 * C_back / dt)
        F_hat = amplitude * noise_complex


        F = np.real(np.fft.ifft2(F_hat))
        return F
