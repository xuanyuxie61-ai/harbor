"""
waveform_migration.py -- Seismic Waveform Migration and Stacking
=================================================================
Implements waveform migration and stacking for earthquake detection
and location in a 3D spatial grid, with UQ on traveltime uncertainties.

The method:
1. Precompute traveltime lookup tables (LUT) from each grid point to stations
2. For each time sample, migrate (shift) waveforms by traveltimes
3. Stack (sum) migrated waveforms to get coalescence map
4. Detect events when coalescence exceeds threshold (STA/LTA)
5. Locate events at the grid point with maximum coalescence

Seed references:
  - 1094_QuakeMigrate_manuscript: detection, location, LUT, STA/LTA, wavelet
  - 1344_triangulation_orient: 3D grid orientation
"""
import numpy as np
from typing import Tuple


class GaussianDerivativeWavelet:
    """
    Gaussian derivative wavelet for synthetic seismogram generation.

    w(t) = -2 * alpha * (t - t0) * exp(-alpha * (t - t0)^2)
    where alpha controls the center frequency and t0 is the onset time.

    The Fourier transform peaks at frequency f_0 = sqrt(alpha) / pi.
    """

    def __init__(self, center_freq: float = 5.0, duration: float = 2.0,
                 sampling_rate: float = 100.0):
        self.alpha = (np.pi * center_freq) ** 2
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.n_samples = int(duration * sampling_rate)
        self.t = np.arange(self.n_samples) / sampling_rate
        self.t0 = duration / 2.0

    def evaluate(self) -> np.ndarray:
        """Evaluate the wavelet at all time samples."""
        dt = self.t - self.t0
        return -2.0 * self.alpha * dt * np.exp(-self.alpha * dt ** 2)

    def derivative(self) -> np.ndarray:
        """Evaluate the time derivative of the wavelet."""
        dt = self.t - self.t0
        w = self.evaluate()
        return w * (-2.0 * self.alpha * dt) + (-2.0 * self.alpha) * np.exp(
            -self.alpha * dt ** 2)


class TraveltimeLUT:
    """
    Traveltime Look-Up Table for a 3D grid of potential source locations
    to a set of receiver stations.

    Uses a simple 1D velocity model: t = distance / v_avg
    with optional Gaussian perturbation for uncertainty modeling.
    """

    def __init__(self, grid_shape: Tuple[int, int, int],
                 grid_spacing: float, station_coords: np.ndarray,
                 velocity_model: float = 5000.0):
        """
        Parameters
        ----------
        grid_shape : tuple
            (nx, ny, nz) grid dimensions.
        grid_spacing : float
            Grid spacing in meters.
        station_coords : ndarray, shape (n_stations, 3)
            Station coordinates (x, y, z) in meters.
        velocity_model : float
            Average P-wave velocity (m/s).
        """
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.stations = station_coords
        self.n_stations = station_coords.shape[0]
        self.velocity = velocity_model

        # Build grid coordinates
        nx, ny, nz = grid_shape
        self.x = np.arange(nx) * grid_spacing
        self.y = np.arange(ny) * grid_spacing
        self.z = np.arange(nz) * grid_spacing

        # Compute traveltimes
        self.p_times = np.zeros((*grid_shape, self.n_stations))
        self.s_times = np.zeros((*grid_shape, self.n_stations))

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    gp = np.array([self.x[ix], self.y[iy], self.z[iz]])
                    for s in range(self.n_stations):
                        dist = np.linalg.norm(gp - station_coords[s])
                        self.p_times[ix, iy, iz, s] = dist / velocity_model
                        self.s_times[ix, iy, iz, s] = dist / (velocity_model / 1.732)

    def get_traveltime(self, grid_idx: Tuple[int, int, int],
                       station: int, phase: str = 'P') -> float:
        """Get traveltime from grid point to station for given phase."""
        ix, iy, iz = grid_idx
        if phase == 'P':
            return self.p_times[ix, iy, iz, station]
        else:
            return self.s_times[ix, iy, iz, station]

    def add_uncertainty(self, std_time: float = 0.05, seed: int = None):
        """Add Gaussian uncertainty to traveltimes for UQ."""
        rng = np.random.RandomState(seed)
        self.p_times += std_time * rng.randn(*self.p_times.shape)
        self.s_times += std_time * rng.randn(*self.s_times.shape)
        self.p_times = np.maximum(self.p_times, 0.0)
        self.s_times = np.maximum(self.s_times, 0.0)


def compute_stalta(trace: np.ndarray, sta_window: int = 10,
                   lta_window: int = 100) -> np.ndarray:
    """
    Compute the STA/LTA (Short-Term Average / Long-Term Average) onset function.

    STA/LTA(t) = (1/sta_win) sum_{i=t-sta}^{t} x_i^2 /
                 (1/lta_win) sum_{i=t-lta}^{t} x_i^2

    This is the standard triggering function for seismic event detection.

    Parameters
    ----------
    trace : ndarray
        Seismic trace (1D time series).
    sta_window : int
        Short-term average window length (samples).
    lta_window : int
        Long-term average window length (samples).

    Returns
    -------
    stalta : ndarray
        STA/LTA ratio time series.
    """
    n = len(trace)
    x2 = trace ** 2
    stalta = np.zeros(n)

    sta_sum = 0.0
    lta_sum = 0.0

    for i in range(n):
        sta_sum += x2[i]
        lta_sum += x2[i]

        if i >= sta_window:
            sta_sum -= x2[i - sta_window]
        if i >= lta_window:
            lta_sum -= x2[i - lta_window]

        sta_avg = sta_sum / min(i + 1, sta_window)
        lta_avg = lta_sum / min(i + 1, lta_window)

        if lta_avg > 1e-30:
            stalta[i] = sta_avg / lta_avg
        else:
            stalta[i] = 0.0

    return stalta


def migrate_and_stack(waveforms: np.ndarray, lut: TraveltimeLUT,
                      origin_time: float = 0.0,
                      sampling_rate: float = 100.0
                      ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Migrate and stack waveforms over the 3D grid to compute the coalescence
    function at each grid point.

    For each grid point (ix, iy, iz):
        C(ix,iy,iz) = sum_s |waveform_s(t_0 + t_p(ix,iy,iz,s))|

    The detected source location is argmax C.

    Parameters
    ----------
    waveforms : ndarray, shape (n_stations, n_samples)
        Recorded waveforms at each station.
    lut : TraveltimeLUT
        Traveltime lookup table.
    origin_time : float
        Reference origin time (seconds).
    sampling_rate : float
        Sampling rate (Hz).

    Returns
    -------
    coalescence : ndarray, shape grid_shape
        Coalescence value at each grid point.
    max_loc : tuple
        (ix, iy, iz) of maximum coalescence.
    """
    nx, ny, nz = lut.grid_shape
    coalescence = np.zeros((nx, ny, nz))
    dt = 1.0 / sampling_rate

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for s in range(lut.n_stations):
                    tt = lut.p_times[ix, iy, iz, s]
                    sample_idx = int((origin_time + tt) / dt)
                    if 0 <= sample_idx < waveforms.shape[1]:
                        coalescence[ix, iy, iz] += abs(waveforms[s, sample_idx])

    max_idx = np.unravel_index(np.argmax(coalescence), coalescence.shape)
    return coalescence, max_idx


def simulate_waveforms(lut: TraveltimeLUT,
                       source_idx: Tuple[int, int, int],
                       origin_time: float = 1.0,
                       duration: float = 5.0,
                       noise_level: float = 0.1,
                       seed: int = None) -> np.ndarray:
    """
    Generate synthetic waveforms for a source at the given grid location.

    Uses a Gaussian derivative wavelet with traveltime shifts and
    distance-based amplitude attenuation.

    Amplitude attenuation: A(d) = A0 / (1 + d/d_ref)^1.5
    where d is hypocentral distance and d_ref is a reference distance.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration * 100)  # 100 Hz sampling
    waveforms = np.zeros((lut.n_stations, n_samples))

    wavelet = GaussianDerivativeWavelet(center_freq=5.0, duration=1.0,
                                        sampling_rate=100.0)
    wav = wavelet.evaluate()

    sx = lut.x[source_idx[0]]
    sy = lut.y[source_idx[1]]
    sz = lut.z[source_idx[2]]

    d_ref = 1000.0  # reference distance (m)

    for s in range(lut.n_stations):
        tt_p = lut.p_times[source_idx[0], source_idx[1], source_idx[2], s]
        dist = np.linalg.norm(
            np.array([sx, sy, sz]) - lut.stations[s])

        # Attenuation
        amplitude = 1.0 / (1.0 + dist / d_ref) ** 1.5

        # Add noise to traveltime
        tt_p_noisy = tt_p + 0.01 * rng.randn()

        # Place wavelet at arrival time
        onset_sample = int((origin_time + tt_p_noisy) * 100)
        wav_len = len(wav)
        start = max(0, onset_sample)
        end = min(n_samples, onset_sample + wav_len)
        if start < end and onset_sample >= 0:
            wave_start = max(0, -onset_sample)
            waveforms[s, start:end] += amplitude * wav[wave_start:wave_start + (end - start)]

        # Add Gaussian noise
        waveforms[s] += noise_level * rng.randn(n_samples)

    return waveforms


def detect_events(coalescence_trace: np.ndarray, threshold: float = 5.0,
                  min_separation: int = 50) -> list:
    """
    Detect seismic events when the coalescence function exceeds a threshold.

    Uses STA/LTA on the coalescence trace, with a minimum separation
    between detected events to avoid multiple triggers for the same event.

    Returns list of (sample_index, coalescence_value) tuples.
    """
    stalta = compute_stalta(coalescence_trace, sta_window=5, lta_window=50)
    events = []
    last_trigger = -min_separation

    for i in range(len(stalta)):
        if stalta[i] > threshold and (i - last_trigger) >= min_separation:
            events.append((i, float(stalta[i])))
            last_trigger = i

    return events
