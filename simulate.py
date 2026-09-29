"""Stream nominal, isolated-fault, and coherent-event AWS telemetry into MOES-AHEAD."""
from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

API_URL = "http://localhost:8000/api/v1/telemetry/stream"
REQUEST_TIMEOUT_SECONDS = 5.0
WINDOW_DELAY_SECONDS = 0.35

STATIONS: List[Dict[str, object]] = [
    {"id": str(uuid.uuid4()), "name": "AWS-DELHI-01", "lat": 28.6139, "lon": 77.2090},
    {"id": str(uuid.uuid4()), "name": "AWS-DELHI-02", "lat": 28.6250, "lon": 77.2200},
    {"id": str(uuid.uuid4()), "name": "AWS-DELHI-03", "lat": 28.6000, "lon": 77.1900},
    {"id": str(uuid.uuid4()), "name": "AWS-DELHI-04", "lat": 28.6300, "lon": 77.1800},
    {"id": str(uuid.uuid4()), "name": "AWS-DELHI-05", "lat": 28.5900, "lon": 77.2300},
]


def send_matrix_data(
    pressure_modifiers: Optional[Dict[int, float]] = None,
    precip_modifiers: Optional[Dict[int, float]] = None,
    pressure_delta_modifiers: Optional[Dict[int, float]] = None,
    station_order: Optional[List[int]] = None,
) -> None:
    """Generate and post one telemetry window, printing the API's actual result."""
    pressure_modifiers = pressure_modifiers or {}
    precip_modifiers = precip_modifiers or {}
    pressure_delta_modifiers = pressure_delta_modifiers or {}
    order = station_order if station_order is not None else list(range(len(STATIONS)))
    print(f"\n--- Ingestion window {datetime.now().astimezone().strftime('%H:%M:%S %Z')} ---")

    for index in order:
        station = STATIONS[index]
        pressure_modifier = pressure_modifiers.get(index, 0.0)
        precipitation_modifier = precip_modifiers.get(index, 0.0)
        pressure_delta = pressure_delta_modifiers.get(index, random.uniform(-0.25, 0.25))
        payload = {
            "sensor_id": station["id"],
            "lat": station["lat"],
            "lon": station["lon"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ambient_pressure": round(1013.25 + random.uniform(-0.35, 0.35) + pressure_modifier, 2),
            "precipitation_rate": round(max(0.0, random.uniform(0.0, 0.8) + precipitation_modifier), 2),
            "wind_speed": round(random.uniform(5.0, 15.0), 2),
            "radar_reflectivity": round(max(0.0, 15.0 + precipitation_modifier * 1.5), 2),
            "pressure_delta_10m": round(pressure_delta, 2),
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()
            print(
                f"[{station['name']}] pressure={payload['ambient_pressure']:.2f} hPa, "
                f"ΔP/10m={payload['pressure_delta_10m']:+.2f} hPa, "
                f"rain={payload['precipitation_rate']:.2f} mm/hr -> "
                f"{result.get('status', 'UNKNOWN')} "
                f"(coherence={result.get('coherence_score', 'n/a')}; "
                f"{result.get('message', 'No message')})"
            )
        except requests.RequestException as error:
            body = error.response.text[:500] if error.response is not None else ""
            print(f"[{station['name']}] request failed: {error}; response={body}")
        except ValueError as error:
            print(f"[{station['name']}] backend returned invalid JSON: {error}")
        time.sleep(WINDOW_DELAY_SECONDS)


if __name__ == "__main__":
    print("Starting MOES-AHEAD sensor validation stream.")
    print(f"Target: {API_URL}")
    print("Start the API with: uvicorn backend:app --reload")

    print("\n[PHASE 1] Nominal microclimate baseline")
    send_matrix_data()
    send_matrix_data()

    print("\n[PHASE 2] Isolated AWS-DELHI-01 transducer drop")
    send_matrix_data(
        pressure_modifiers={0: -15.0},
        pressure_delta_modifiers={0: -15.0},
    )

    print("\n[PHASE 3] Corroborated regional pressure drop and heavy rainfall")
    # Stream the four peers first so the last station is assessed against a
    # contemporaneous event signature from the local network.
    send_matrix_data(
        pressure_modifiers={0: -12.0, 1: -9.5, 2: -10.2, 3: -8.9, 4: -11.0},
        precip_modifiers={0: 45.0, 1: 38.0, 2: 40.5, 3: 32.0, 4: 42.0},
        pressure_delta_modifiers={0: -12.0, 1: -9.5, 2: -10.2, 3: -8.9, 4: -11.0},
        station_order=[1, 2, 3, 4, 0],
    )
    print("\nSimulation complete. Inspect the API responses and GET /api/v1/alerts/broadcast.")
