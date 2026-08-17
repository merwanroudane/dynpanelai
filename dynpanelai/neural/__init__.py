"""Neural lag-discovery models and their audit protocol (Xu, 2026)."""

from .acgate import ACGate, ACGateResult, build_lag_tensors
from .audit import (
    AuditReport,
    audit_l1,
    audit_l2,
    audit_l3,
    fisher_combine,
    run_audit,
)

__all__ = [
    "ACGate",
    "ACGateResult",
    "build_lag_tensors",
    "audit_l1",
    "audit_l2",
    "audit_l3",
    "fisher_combine",
    "run_audit",
    "AuditReport",
]
