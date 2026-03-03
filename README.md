# SN-CAST 2.0

The Seismic Network Capability Assessment Software Tool

This version of SN-CAST is updated from the original code supporting Möllhoff et al., (2019).

## Overview

SN-CAST (Seismic Network Capability Assessment Software Tool) is a Python package for modeling and assessing the detection capability of seismic networks. It calculates minimum detectable magnitudes across a geographic grid based on station noise levels, network geometry, and various ground motion prediction models.

## Features

- Network detection capability modeling using local magnitude (ML) or GMPE methods
- Support for multiple types on instrumentation: traditional stations, seismic arrays, ocean bottom seismometers (OBS), and distributed acoustic sensing (DAS).
- Support for  UK and California empirical local magnitude scales.
- Flexible grid-based calculations with multiprocessing support
- Noise estimation tools for seismic data
- **Cross-section analysis** for 2D detection capability profiles

### Work in progress

- Support for using ground motion prediction equations (GMPEs) instead of local magnitude scales

## Installation

### Requirements

See pyproject.toml

### Install from source

```bash
git clone https://github.com/jasplet/sncast.git
cd sncast
pip install -e .
```

**PiPy release TBD**

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic usage

```python
from sncast.model_detection_capability import find_min_ml

# Calculate minimum detectable magnitude for a grid
result = find_min_ml(
    lon0=-1.0, lon1=1.0,      # Longitude bounds
    lat0=50.0, lat1=52.0,      # Latitude bounds
    dlon=0.2, dlat=0.2,        # Grid spacing
    networks=['stations.csv'], # Station data file
    stat_num=[5],              # Number of stations required
    foc_depth=0.0,             # Focal depth in km
    snr=3.0,                   # Signal-to-noise ratio
    mag_min=-2.0,              # Minimum magnitude to test
    mag_delta=0.1,             # Magnitude increment
    method='ML',               # Method: 'ML' or 'GMPE'
    region='UK'                # Region: 'UK' or 'CAL'
)

# Result is an xarray DataArray with detection capability grid
print(result)
```

### Station data format

Station data should be a CSV file with the following columns:

```csv
station,longitude,latitude,elevation_km,noise [nm]
STA1,0.0,50.0,0.0,1.0
STA2,0.5,50.5,0.1,1.5
STA3,1.0,51.0,0.2,2.0
```

### DAS data format

```csv
channel_index,fiber_length_m,longitude,latitude,noise_m,elevation_km
10010,1000,0.01,50.01,1e-8,0.0
10020,2000,0.02,50.02,2e-8,0.0
```

## Documentation

Full documentation is under construction.

### Key modules

- **`model_detection_capability`**: Main detection modeling functions
- **`gmpes`**: Ground motion prediction equations
- **`magnitude_conversions`**: ML ↔ Mw conversions
- **`noise_estimation`**: Seismic noise analysis tools

## Examples

See the `examples/` directory for:

- Basic network modeling
- Multi-network analysis (stations + arrays + DAS)
- Cross-section calculations
- Custom GMPE usage

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=sncast --cov-report=html
```

## Citation

If you use SN-CAST in your research, please cite:

**Original version:**
```
Möllhoff, M., Bean, C. J., & Meredith, P. G. (2019). Seismic detection 
capability of the Irish National Seismic Network. Seismological Research 
Letters, 90(4), 1607-1617.
```

**Version 2.0:**
```
[Your citation here when published]
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

