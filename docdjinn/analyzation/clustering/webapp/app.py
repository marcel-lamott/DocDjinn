import dash
import dash_bootstrap_components as dbc

from docdjinn.analyzation.clustering.webapp.config import settings
from docdjinn.analyzation.clustering.webapp.components import create_app_layout
from docdjinn.analyzation.clustering.webapp.callbacks import register_callbacks
from docdjinn.analyzation.clustering.webapp.server_routes import setup_server_routes
from docdjinn.analyzation.clustering.webapp.data_manager import data_manager


def create_app():
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Setup Flask server routes
    setup_server_routes(app.server)

    # Load initial dataset
    data_manager.load_dataset(settings.default_dataset)

    # Set layout
    app.layout = create_app_layout()

    # Register callbacks
    register_callbacks(app)

    return app


def main():
    """Main entry point."""
    app = create_app()
    app.run(debug=settings.debug, port=settings.port)


if __name__ == "__main__":
    main()
