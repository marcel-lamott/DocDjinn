"""
Document clustering visualization web application.

This package provides an interactive Dash web application for visualizing
document clustering results with scatter plots, cluster analysis, and
document preview capabilities.
"""

from .app import create_app, main

__all__ = ["create_app", "main"]
