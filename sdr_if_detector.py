#!/usr/bin/env python3
"""
sdr_if_detector.py
------------------
Detect intermediate frequencies (IF) that may originate from >20 GHz telemetry radars
using a commodity SDR + downconverter (or front-ends with built-in IF outputs).

Why IF?
- Many radars (e.g., 24/28/37–40/60/77–81 GHz) are downconverted to IFs like 10.7/21.4/70/140/240 MHz or L-band.
- Commodity SDRs (RTL-SDR, HackRF, etc.) can monitor these IFs even if they cannot directly tune to GHz ranges.

What this tool does
- Sweeps user-specified IF ranges and common IF "hotspots"
- Computes PSD in real-time, adapts a noise floor model, and flags statistically significant peaks
- Logs verifiable events (timestamped JSON/CSV) and optionally saves IQ/PSD snapshots for evidence
- Optional continuous watch mode with alerts printed to console

Supported SDR backends
- SoapySDR (recommended; supports many radios)
- pyrtlsdr (RTL2832U-based dongles)
- If neither is available, runs in SIM mode generating synthetic peaks (for testing pipelines)

Usage (examples)
- Basic sweep with defaults:
    python sdr_if_detector.py --device auto --out out/
- Sweep specific IFs (MHz) around ±2 MHz:
    python sdr_if_detector.py --centers 10.7,21.4,70,140,240 --span 4 --sr 2.4
- Continuous watch on 140 MHz ± 5 MHz, stronger detection:
    python sdr_if_detector.py --centers 140 --span 10 --watch --threshold_db 10

Dependencies
- numpy, scipy, click; optional: SoapySDR or pyrtlsdr, sounddevice (for future alert tones)

DISCLAIMER
- This tool is for defensive spectrum monitoring, research, and compliance.
- Always comply with local laws/regulations when monitoring RF environments.
- Detection ≠ attribution; follow scientific and legal best practices.
"""

import os, sys, time, math, json, datetime, pathlib, csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal
import click

# Backend selection -----------------------------------------------------------
_BACKEND = None
try:
    import SoapySDR  # type: ignore
    _BACKEND = 'soapy'
except Exception:
    try:
        from rtlsdr import RtlSdr  # type: ignore
        _BACKEND = 'rtlsdr'
    except Exception:
        _BACKEND = 'sim'  # no SDR libs available

# Utils ----------------------------------------------------------------------
def db10(x):
    x = np.maximum(x, 1e-20)
    return 10.0 * np.log10(x)

def median_mad_db(psd_db):
    med = np.median(psd_db)
    mad = np.median(np.abs(psd_db - med)) * 1.4826  # robust sigma
    return med, mad

def welch_psd(iq, fs, nperseg=4096, noverlap=2048):
    f, pxx = signal.welch(iq, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False, scaling='density')
    # Shift to baseband center-first ordering
    idx = np.argsort(f)
    f = f[idx]
    pxx = pxx[idx]
    return f, pxx

@dataclass
class DetectionEvent:
    timestamp_utc: str
    center_mhz: float
    span_mhz: float
    peak_freq_mhz: float
    peak_power_db: float
    noise_floor_db: float
    snr_db: float
    bandwidth_khz: float
    backend: str
    sample_rate_msps: float
    gain: Optional[float] = None
    notes: Optional[str] = None

# SDR interfaces -------------------------------------------------------------
class SDRBase:
    def open(self): ...
    def close(self): ...
    def configure(self, center_hz:int, sample_rate_hz:int, gain:Optional[float]): ...
    def read_samples(self, num:int)->np.ndarray: ...

class SDRSim(SDRBase):
    def __init__(self, seed=1337):
        self.rng = np.random.default_rng(seed)
        self.fs = 2_400_000
        self.fc = 100_000_000
        self.gain = None
    def open(self): pass
    def close(self): pass
    def configure(self, center_hz:int, sample_rate_hz:int, gain:Optional[float]):
        self.fc = center_hz; self.fs = sample_rate_hz; self.gain = gain
    def read_samples(self, num:int)->np.ndarray:
        t = np.arange(num)/self.fs
        noise = (self.rng.normal(size=num) + 1j*self.rng.normal(size=num)) * 0.1
        # inject 1-2 synthetic tones
        f1 = (self.rng.uniform(-0.3,0.3))*self.fs*0.5
        tone = np.exp(2j*np.pi*f1*t)*0.7
        return noise + tone

class SDRSoapy(SDRBase):
    def __init__(self, args=""):
        self.args = args
        self.dev = None
        self.fs = None
        self.fc = None
        self.gain = None
    def open(self):
        self.dev = SoapySDR.Device(self.args if self.args else {})
        self.dev.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2_400_000)
        self.dev.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, 2_400_000)
        self.dev.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "RX")
        self.dev.activateStream(self.dev.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32))
    def close(self):
        # Best-effort cleanup
        try:
            self.dev.deactivateStream(self.dev.getStream(SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            pass
    def configure(self, center_hz:int, sample_rate_hz:int, gain:Optional[float]):
        self.dev.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate_hz)
        self.dev.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, float(center_hz))
        if gain is not None:
            self.dev.setGain(SoapySDR.SOAPY_SDR_RX, 0, float(gain))
        self.fs = sample_rate_hz; self.fc = center_hz; self.gain = gain
    def read_samples(self, num:int)->np.ndarray:
        buff = np.zeros(num, dtype=np.complex64)
        sr = self.dev.getStream(SoapySDR.SOAPY_SDR_RX, 0)
        # Read loop (simplified)
        total = 0
        while total < num:
            # Soapy readStream signature may differ; this is indicative.
            n = self.dev.readStream(sr, [buff[total:]], num-total)
            if isinstance(n, tuple):
                n = n[0]
            if n > 0: total += n
        return buff

class SDRRTL(SDRBase):
    def __init__(self):
        from rtlsdr import RtlSdr
        self.dev = RtlSdr()
        self.fs = None; self.fc = None; self.gain = None
    def open(self): pass
    def close(self): 
        try: self.dev.close()
        except Exception: pass
    def configure(self, center_hz:int, sample_rate_hz:int, gain:Optional[float]):
        self.dev.sample_rate = sample_rate_hz
        self.dev.center_freq = center_hz
        if gain is not None:
            self.dev.gain = gain
        self.fs = sample_rate_hz; self.fc = center_hz; self.gain = gain
    def read_samples(self, num:int)->np.ndarray:
        return self.dev.read_samples(num).astype(np.complex64)

def get_sdr(device:str):
    if device == 'auto':
        if _BACKEND == 'soapy': return SDRSoapy()
        if _BACKEND == 'rtlsdr': return SDRRTL()
        return SDRSim()
    if device == 'soapy': return SDRSoapy()
    if device == 'rtlsdr': return SDRRTL()
    return SDRSim()

# Detection logic ------------------------------------------------------------
def detect_peaks(psd_db, freqs_hz, k_sigma=6.0, min_bw_bins=2):
    """Return list of (f_peak_hz, p_peak_db, bw_khz, snr_db, noise_db)."""
    med, mad = median_mad_db(psd_db)
    thr = med + k_sigma * mad
    peaks, _ = signal.find_peaks(psd_db, height=thr)
    events = []
    for p in peaks:
        # bandwidth estimate via -3 dB points around peak
        peak_db = psd_db[p]
        left = p
        while left > 0 and psd_db[left] > peak_db - 3:
            left -= 1
        right = p
        while right < len(psd_db)-1 and psd_db[right] > peak_db - 3:
            right += 1
        if right-left+1 < min_bw_bins:
            continue
        bw_hz = freqs_hz[right] - freqs_hz[left]
        fpk = freqs_hz[p]
        snr = peak_db - med
        events.append((float(fpk), float(peak_db), float(bw_hz/1e3), float(snr), float(med)))
    return events

def run_sweep(sdr:SDRBase, centers_mhz:List[float], span_mhz:float, sample_rate_msps:float,
              gain:Optional[float], seconds_per_center:float, threshold_sigma:float,
              out_dir:str, watch:bool, save_iq:bool, project:str):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_csv = os.path.join(out_dir, "events.csv")
    first_csv = not os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if first_csv:
            writer.writerow(["timestamp_utc","center_mhz","span_mhz","peak_freq_mhz","peak_power_db",
                             "noise_floor_db","snr_db","bandwidth_khz","backend","sample_rate_msps","gain","notes"])
        sdr.open()
        try:
            while True:
                for c in centers_mhz:
                    fc = int(c*1e6)
                    fs = int(sample_rate_msps*1e6)
                    span_hz = int(span_mhz*1e6)
                    # Ensure fs >= span + guard
                    if fs < span_hz*1.25:
                        fs = int(span_hz*1.5)
                    sdr.configure(fc, fs, gain)
                    N = int(fs * seconds_per_center)
                    iq = sdr.read_samples(N)
                    freqs, pxx = welch_psd(iq, fs=fs, nperseg=4096, noverlap=2048)
                    psd_db = db10(pxx)
                    events = detect_peaks(psd_db, freqs, k_sigma=threshold_sigma, min_bw_bins=3)
                    ts = datetime.datetime.utcnow().isoformat()+"Z"
                    for (fpk, p_db, bw_khz, snr, noise_db) in events:
                        abs_freq_hz = fc + fpk
                        row = DetectionEvent(
                            timestamp_utc=ts,
                            center_mhz=float(fc/1e6),
                            span_mhz=span_mhz,
                            peak_freq_mhz=float(abs_freq_hz/1e6),
                            peak_power_db=p_db,
                            noise_floor_db=noise_db,
                            snr_db=snr,
                            bandwidth_khz=bw_khz,
                            backend=_BACKEND,
                            sample_rate_msps=float(fs/1e6),
                            gain=gain,
                            notes=project
                        )
                        writer.writerow([row.timestamp_utc,row.center_mhz,row.span_mhz,row.peak_freq_mhz,
                                         row.peak_power_db,row.noise_floor_db,row.snr_db,row.bandwidth_khz,
                                         row.backend,row.sample_rate_msps,row.gain,row.notes])
                        if save_iq:
                            out_iq = os.path.join(out_dir, f"iq_{ts}_{int(abs_freq_hz)}.npy")
                            np.save(out_iq, iq)
                    # console alert
                    if events:
                        print(f"[{ts}] {len(events)} event(s) near center {c:.3f} MHz")
                        for e in events:
                            print(f"  - Peak @ {c + e[0]/1e6:+.3f} MHz, SNR {e[3]:.1f} dB, BW {e[2]:.1f} kHz")
                if not watch:
                    break
        finally:
            sdr.close()

# CLI ------------------------------------------------------------------------
COMMON_IFS_MHZ = [10.7, 21.4, 45.0, 70.0, 110.0, 140.0, 240.0, 380.0, 1400.0]

@click.command()
@click.option("--device", type=click.Choice(["auto","soapy","rtlsdr","sim"]), default="auto", show_default=True,
              help="SDR backend selection")
@click.option("--centers", type=str, default=",".join(str(x) for x in COMMON_IFS_MHZ),
              help="Comma-separated IF centers in MHz to scan")
@click.option("--span", type=float, default=4.0, show_default=True, help="Span (MHz) around each center")
@click.option("--sr", type=float, default=2.4, show_default=True, help="Sample rate MS/s")
@click.option("--gain", type=float, default=None, help="Frontend gain (dB) if supported")
@click.option("--seconds", type=float, default=2.0, show_default=True, help="Seconds per center")
@click.option("--threshold_db", type=float, default=6.0, show_default=True,
              help="Detection threshold in sigma (robust) over noise floor")
@click.option("--out", "out_dir", type=click.Path(), default="out", show_default=True, help="Output directory")
@click.option("--watch", is_flag=True, help="Continuous sweep/watch mode")
@click.option("--save-iq", is_flag=True, help="Save raw IQ snapshots for flagged events")
@click.option("--project", type=str, default="waveless-world", show_default=True, help="Tag saved in logs")
def main(device, centers, span, sr, gain, seconds, threshold_db, out_dir, watch, save_iq, project):
    """
    SDR IF Detector for telemetry radar IFs.
    """
    centers_mhz = [float(x.strip()) for x in centers.split(",") if x.strip()]
    sdr = get_sdr(device)
    print(f"Backend: {_BACKEND} | Centers: {centers_mhz} MHz | Span: ±{span/2:.2f} MHz | SR: {sr} MS/s")
    print(f"Output → {out_dir} | Watch: {watch} | Threshold: {threshold_db}σ")
    run_sweep(sdr, centers_mhz, span, sr, gain, seconds, threshold_db, out_dir, watch, save_iq, project)

if __name__ == "__main__":
    main()
