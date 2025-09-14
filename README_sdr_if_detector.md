# SDR IF Detector (Telemetry Radar IFs)

This Python tool scans likely **intermediate frequencies (IF)** associated with >20 GHz telemetry radars (e.g., 24/28/37–40/60/77–81 GHz) using commodity SDRs and downconverters.

## Features
- Welch PSD, robust noise floor, sigma-threshold peak detection
- Default IF centers: 10.7, 21.4, 45, 70, 110, 140, 240, 380, 1400 MHz (editable)
- CSV logging of detections; optional IQ snapshot saving
- Works with **SoapySDR** or **pyrtlsdr**; falls back to **SIM** mode for testing

## Quick start
```bash
pip install numpy scipy click SoapySDR pyrtlsdr
python sdr_if_detector.py --device auto --out out
```

## Notes
- Tune centers/spans to match your downconverter or IF output.
- Detection != attribution; use evidence procedures for legal use.
