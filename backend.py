from __future__ import annotations
import asyncio
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("moes_ahead.anomaly")

class TelemetryIngestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensor_id: uuid.UUID
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timestamp: datetime
    ambient_pressure: float = Field(ge=850, le=1080, description="hPa")
    precipitation_rate: float = Field(ge=0, le=1000, description="mm/hr")
    wind_speed: float = Field(ge=0, le=300, description="knots")
    radar_reflectivity: float = Field(ge=-40, le=100, description="dBZ")
    pressure_delta_10m: Optional[float] = Field(default=None, ge=-100, le=100, description="hPa over ten minutes")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

class ValidationOutcome(str, Enum):
    VERIFIED = "VERIFIED"
    SENSOR_ISOLATION_SUPPRESSED = "SENSOR_ISOLATION_SUPPRESSED"
    NOMINAL = "NOMINAL"
    INSUFFICIENT_PEERS = "INSUFFICIENT_PEERS"

class SystemStatusResponse(BaseModel):
    status: ValidationOutcome
    is_valid: bool
    coherence_score: float = Field(ge=0, le=1)
    message: str
    sensor_id: uuid.UUID
    evaluated_at: datetime

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))

class SpatialCoherenceValidator:
    def __init__(self, spatial_lookup_radius_km: float = 15.0, mahalanobis_alpha: float = 0.01):
        if spatial_lookup_radius_km <= 0 or not 0 < mahalanobis_alpha < 1:
            raise ValueError("radius must be positive and alpha must be between zero and one")
        self.spatial_lookup_radius_km = spatial_lookup_radius_km
        self.mahalanobis_alpha = mahalanobis_alpha

    def compute_spatial_coherence(self, target: TelemetryIngestPayload,
                                  peer_network: List[TelemetryIngestPayload]) -> Tuple[bool, float, str]:
        logger.info("coherence.start sensor=%s peers=%d", target.sensor_id, len(peer_network))
        local = [(p, haversine_km(target.lat, target.lon, p.lat, p.lon))
                 for p in peer_network if p.sensor_id != target.sensor_id]
        local = [(p, d) for p, d in local if d <= self.spatial_lookup_radius_km]
        if len(local) < 2:
            logger.warning("coherence.insufficient sensor=%s local_peers=%d", target.sensor_id, len(local))
            return True, 0.5, "INSUFFICIENT_PEERS: retained pending corroboration"
        peers = [p for p, _ in local]
        X = np.array([[p.ambient_pressure, p.precipitation_rate, p.wind_speed, p.radar_reflectivity] for p in peers], dtype=float)
        y = np.array([target.ambient_pressure, target.precipitation_rate, target.wind_speed, target.radar_reflectivity], dtype=float)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) if len(X) > 2 else np.diag(np.maximum(X.var(axis=0), 1e-5))
        cov = np.atleast_2d(cov) + np.eye(4) * 1e-5
        delta = y - mean
        distance = float(np.sqrt(max(0.0, delta @ np.linalg.pinv(cov) @ delta)))
        scales = np.sqrt(np.maximum(np.diag(cov), 1e-5))
        z = delta / scales
        pressure_drop = target.pressure_delta_10m is not None and target.pressure_delta_10m <= -4.0
        rain_spike = target.precipitation_rate >= max(20.0, float(np.quantile(X[:, 1], .75) + 2 * max(float(np.std(X[:, 1])), 1.0)))
        event = pressure_drop or rain_spike
        peers_direction = [p.pressure_delta_10m for p in peers if p.pressure_delta_10m is not None]
        aligned = len(peers_direction) >= 2 and sum(v < 0 for v in peers_direction) / len(peers_direction) >= .6
        rain_aligned = sum(p.precipitation_rate > max(2.0, float(np.median(X[:, 1]))) for p in peers) >= 2
        corroborated = aligned or rain_aligned
        logger.info("coherence.metrics sensor=%s radius_peers=%d mahalanobis=%.3f pressure_z=%.3f rain_z=%.3f event=%s aligned=%s",
                    target.sensor_id, len(peers), distance, z[0], z[1], event, corroborated)
        if event and corroborated:
            score = float(np.clip(1.0 - distance / 20.0, .7, 1.0))
            return True, score, "VERIFIED: Micro-Regional Extreme Event (Anomalous True Positive)"
        if event and not corroborated and distance >= 3.0:
            return False, float(np.clip(1.0 - distance / 20.0, 0, .49)), "SENSOR_ISOLATION_SUPPRESSED: Hardware Transducer Failure / Uncalibrated Drift"
        score = float(np.clip(1.0 - distance / 20.0, 0, 1))
        return True, score, "NOMINAL: measurement consistent with spatial peer network"

class SpatioTemporalGNNRegressor(torch.nn.Module):
    def __init__(self, input_features: int = 4, hidden_size: int = 32, mesh_height: int = 12,
                 mesh_width: int = 12, forecast_steps: int = 13):
        super().__init__()
        self.mesh_height, self.mesh_width, self.forecast_steps = mesh_height, mesh_width, forecast_steps
        self.input_projection = torch.nn.Linear(input_features, hidden_size)
        self.temporal = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mesh_head = torch.nn.Linear(hidden_size, mesh_height * mesh_width)
        self.step_bias = torch.nn.Parameter(torch.linspace(.15, -.3, forecast_steps))

    def forward(self, adjacency: torch.Tensor, temporal_history: torch.Tensor) -> torch.Tensor:
        if temporal_history.ndim != 3 or adjacency.ndim != 2:
            raise ValueError("temporal_history must be [sensors,time,features], adjacency must be [sensors,sensors]")
        nodes, _, _ = temporal_history.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency dimensions must match sensor count")
        adjacency = adjacency.to(device=temporal_history.device, dtype=temporal_history.dtype)
        adjacency = adjacency.clamp_min(0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        features = torch.relu(self.input_projection(temporal_history))
        features = torch.einsum("ij,jth->ith", adjacency, features)
        _, hidden = self.temporal(features)
        regional_state = hidden[-1].mean(dim=0)
        base_mesh = self.mesh_head(regional_state).reshape(self.mesh_height, self.mesh_width)
        logits = base_mesh.unsqueeze(0) + self.step_bias[:, None, None]
        return torch.sigmoid(logits)

class CAPArea(BaseModel):
    areaDesc: str
    polygon: str

class CAPInfo(BaseModel):
    category: List[str] = ["Met"]
    event: str
    urgency: str
    severity: str
    certainty: str
    headline: str
    description: str
    area: List[CAPArea]

class CAPAlert(BaseModel):
    identifier: str
    sender: str
    sent: datetime
    status: str = "Actual"
    msgType: str = "Alert"
    scope: str = "Public"
    info: List[CAPInfo]

app = FastAPI(title="SIH26077 - Hyper-Local Nowcasting Core", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
validator = SpatialCoherenceValidator()
cache: Dict[uuid.UUID, TelemetryIngestPayload] = {}
cache_lock = asyncio.Lock()
alert_lock = asyncio.Lock()
active_alerts: List[CAPAlert] = []
triggered_alerts: int = 0

@app.post("/api/v1/telemetry/stream", response_model=SystemStatusResponse)
async def ingest_telemetry(payload: TelemetryIngestPayload) -> SystemStatusResponse:
    logger.info("ingest.received sensor=%s timestamp=%s", payload.sensor_id, payload.timestamp.isoformat())
    async with cache_lock:
        peers = list(cache.values())
    is_valid, score, detail = validator.compute_spatial_coherence(payload, peers)
    status = (ValidationOutcome.VERIFIED if detail.startswith("VERIFIED") else
              ValidationOutcome.SENSOR_ISOLATION_SUPPRESSED if not is_valid else
              ValidationOutcome.INSUFFICIENT_PEERS if detail.startswith("INSUFFICIENT") else ValidationOutcome.NOMINAL)
    async with cache_lock:
        cache[payload.sensor_id] = payload
        if len(cache) > 5000:
            oldest_id = min(cache, key=lambda key: cache[key].timestamp)
            cache.pop(oldest_id, None)
    if status == ValidationOutcome.VERIFIED:
        logger.warning("alert.localized_extreme_verified sensor=%s", payload.sensor_id)
        await build_and_store_alert(payload)
    return SystemStatusResponse(status=status, is_valid=is_valid, coherence_score=score,
                                message=detail, sensor_id=payload.sensor_id, evaluated_at=datetime.now(timezone.utc))

async def build_and_store_alert(payload: TelemetryIngestPayload) -> None:
    global triggered_alerts
    stamp = datetime.now(timezone.utc)
    north, west, south, east = payload.lat + .045, payload.lon - .055, payload.lat - .045, payload.lon + .055
    polygon = f"{north:.5f},{west:.5f} {north:.5f},{east:.5f} {south:.5f},{east:.5f} {south:.5f},{west:.5f} {north:.5f},{west:.5f}"
    alert = CAPAlert(identifier=f"MOES-{payload.sensor_id}-{int(stamp.timestamp())}", sender="moes-ahead@mausam.gov.in",
                     sent=stamp, info=[CAPInfo(event="Localized Extreme Rainfall", urgency="Immediate",
                     severity="Severe", certainty="Likely", headline="Verified micro-regional weather extreme",
                     description="Spatial peer corroboration verified a localized severe weather event. Automated dissemination is active.",
                     area=[CAPArea(areaDesc="Affected local sub-district", polygon=polygon)])])
    async with alert_lock:
        active_alerts.insert(0, alert)
        del active_alerts[100:]
        triggered_alerts += 1
    logger.info("alert.cap_compiled id=%s cell_broadcast=queued", alert.identifier)

@app.get("/api/v1/alerts/broadcast", response_model=List[CAPAlert])
async def broadcast_alerts() -> List[CAPAlert]:
    async with alert_lock:
        return list(active_alerts)

@app.get("/api/v1/system/health")
async def system_health() -> dict:
    async with cache_lock:
        count = len(cache)
    return {"status": "Dual-Node Anomaly Engine Active", "ingested_sensors": count,
            "verified_alerts": triggered_alerts, "timestamp": datetime.now(timezone.utc).isoformat()}
