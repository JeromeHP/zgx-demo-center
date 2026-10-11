"""Outbound research clients. Every request here goes through app/egress.py."""

from .grants_gov import GrantsGovClient, Opportunity
from .web import WebResearch

__all__ = ["GrantsGovClient", "Opportunity", "WebResearch"]
