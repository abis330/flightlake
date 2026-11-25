"""Hybrid-electric STOL simulator with ground-truth fault injection."""
from .stol import FaultSpec, SimConfig, simulate
from .writer import generate_fleet, write_flight

__all__ = ["FaultSpec", "SimConfig", "simulate", "generate_fleet", "write_flight"]
