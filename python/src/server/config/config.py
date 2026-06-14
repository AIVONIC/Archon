"""
Environment configuration management for the MCP server.
"""

import os
from dataclasses import dataclass


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""

    pass


@dataclass
class EnvironmentConfig:
    """Configuration loaded from environment variables."""

    database_url: str
    port: int  # Required - no default
    openai_api_key: str | None = None
    host: str = "0.0.0.0"
    transport: str = "sse"


@dataclass
class RAGStrategyConfig:
    """Configuration for RAG strategies."""

    use_contextual_embeddings: bool = False
    use_hybrid_search: bool = True
    use_agentic_rag: bool = True
    use_reranking: bool = True


@dataclass
class MCPMonitoringConfig:
    """Configuration for MCP server monitoring strategy.

    Controls how archon-server monitors MCP server status - via HTTP health checks
    (secure, default) or Docker socket (legacy, security risk).

    Attributes:
        enable_docker_socket: Whether to use Docker socket for container status.
                            Default False for security (uses HTTP health checks).
        health_check_timeout: Timeout in seconds for HTTP health check requests.
    """

    enable_docker_socket: bool = False
    health_check_timeout: int = 5


def validate_openai_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    if not api_key:
        raise ConfigurationError("OpenAI API key cannot be empty")

    if not api_key.startswith("sk-"):
        raise ConfigurationError("OpenAI API key must start with 'sk-'")

    return True


def validate_openrouter_api_key(api_key: str) -> bool:
    """Validate OpenRouter API key format."""
    if not api_key:
        raise ConfigurationError("OpenRouter API key cannot be empty")

    if not api_key.startswith("sk-or-v1-"):
        raise ConfigurationError(
            "OpenRouter API key must start with 'sk-or-v1-'. " "Get your key at https://openrouter.ai/keys"
        )

    return True


def load_environment_config() -> EnvironmentConfig:
    """Load and validate environment configuration."""
    # OpenAI API key is optional at startup - can be set via API
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Required environment variable for database access
    database_url = os.getenv("ARCHON_DATABASE_URL")
    if not database_url:
        raise ConfigurationError("ARCHON_DATABASE_URL environment variable is required")

    # Validate required fields
    if openai_api_key:
        validate_openai_api_key(openai_api_key)

    # Optional environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT")
    if not port_str:
        # This appears to be for MCP configuration based on default 8051
        port_str = os.getenv("ARCHON_MCP_PORT")
        if not port_str:
            raise ConfigurationError(
                "PORT or ARCHON_MCP_PORT environment variable is required. "
                "Please set it in your .env file or environment. "
                "Default value: 8051"
            )
    transport = os.getenv("TRANSPORT", "sse")

    # Validate and convert port
    try:
        port = int(port_str)
    except ValueError as e:
        raise ConfigurationError(f"PORT must be a valid integer, got: {port_str}") from e

    return EnvironmentConfig(
        openai_api_key=openai_api_key,
        database_url=database_url,
        host=host,
        port=port,
        transport=transport,
    )


def get_config() -> EnvironmentConfig:
    """Get environment configuration with validation."""
    return load_environment_config()


def get_rag_strategy_config() -> RAGStrategyConfig:
    """Load RAG strategy configuration from environment variables."""

    def str_to_bool(value: str | None) -> bool:
        """Convert string environment variable to boolean."""
        if value is None:
            return False
        return value.lower() in ("true", "1", "yes", "on")

    return RAGStrategyConfig(
        use_contextual_embeddings=str_to_bool(os.getenv("USE_CONTEXTUAL_EMBEDDINGS")),
        use_hybrid_search=str_to_bool(os.getenv("USE_HYBRID_SEARCH")),
        use_agentic_rag=str_to_bool(os.getenv("USE_AGENTIC_RAG")),
        use_reranking=str_to_bool(os.getenv("USE_RERANKING")),
    )


def get_mcp_monitoring_config() -> MCPMonitoringConfig:
    """Load MCP monitoring configuration from environment variables.

    Environment Variables:
        ENABLE_DOCKER_SOCKET_MONITORING: "true"/"false" (default: false)
            Controls whether to use Docker socket for status monitoring.
            Default is false for security (uses HTTP health checks instead).
        MCP_HEALTH_CHECK_TIMEOUT: Timeout in seconds (default: 5)
            Timeout for HTTP health check requests to MCP server.

    Returns:
        MCPMonitoringConfig with parsed settings.
    """

    def str_to_bool(value: str | None) -> bool:
        """Convert string environment variable to boolean."""
        if value is None:
            return False
        return value.lower() in ("true", "1", "yes", "on")

    return MCPMonitoringConfig(
        enable_docker_socket=str_to_bool(os.getenv("ENABLE_DOCKER_SOCKET_MONITORING")),
        health_check_timeout=int(os.getenv("MCP_HEALTH_CHECK_TIMEOUT", "5")),
    )
