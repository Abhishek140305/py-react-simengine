# MOES-AHEAD Hyper-Local Nowcasting Core

## Dashboard

App.jsx is a React dashboard component and App.css contains its dark aviation operations styling. Import App into a React application entry point. The interface uses in-file mock station telemetry and does not require image or map services.

## API

Install the Python requirements, then start the FastAPI service from this directory:

    python -m pip install -r requirements.txt
    uvicorn backend:app --reload

The interactive API schema is available at /docs. Main routes:

- POST /api/v1/telemetry/stream validates telemetry and performs local spatial divergence analysis.
- GET /api/v1/alerts/broadcast returns stored CAP v1.2-shaped alerts after verified regional extremes.
- GET /api/v1/system/health reports cache and alert counts.

## Sensor validation simulator

With the API running, execute `python simulate.py`. It streams two nominal windows, an isolated AWS-DELHI-01 pressure transducer failure, and a peer-corroborated regional drop with heavy precipitation. The script prints the backend's actual status and coherence score for every station. It uses stable sensor UUIDs for each run and requires the `requests` dependency listed in requirements.txt.

Telemetry JSON fields are sensor_id (UUID), lat, lon, timezone-aware timestamp, ambient_pressure, precipitation_rate, wind_speed, radar_reflectivity, and optional pressure_delta_10m.
