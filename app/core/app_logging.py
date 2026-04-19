import logging
from urllib.parse import urlparse


LOGGER_NAME = "ultimate_macroeconomics_dashboard"
DEFAULT_LOG_FILE_NAME = "app.log"


def _normalize_text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def get_app_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_ultimate_logger_configured", False):
        return logger

    log_path = f"_container_data/{DEFAULT_LOG_FILE_NAME}"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    logger._ultimate_logger_configured = True
    return logger


def log_page_render(page_name: str) -> None:
    get_app_logger().info("page_render page=%s", _normalize_text(page_name, limit=120))


def log_sql_query(query: str, target: str = "postgres_db") -> None:
    get_app_logger().info(
        "sql_query target=%s query=%s",
        _normalize_text(target, limit=120),
        _normalize_text(query),
    )


def log_http_request(
    base_url: str | None,
    endpoint: str,
    method: str,
    summary: str | None = None,
) -> None:
    parsed = urlparse(str(base_url or "").strip())
    target = parsed.netloc or parsed.path or "unknown"

    get_app_logger().info(
        "http_request target=%s method=%s endpoint=%s summary=%s",
        _normalize_text(target, limit=120),
        _normalize_text(method.upper(), limit=16),
        _normalize_text(endpoint, limit=80),
        _normalize_text(summary or "-"),
    )


def log_vector_query(
    operation: str,
    collection_name: str | None = None,
    summary: str | None = None,
    target: str = "vector_db",
) -> None:
    get_app_logger().info(
        "vector_query target=%s operation=%s collection=%s summary=%s",
        _normalize_text(target, limit=120),
        _normalize_text(operation, limit=80),
        _normalize_text(collection_name or "-", limit=160),
        _normalize_text(summary or "-"),
    )
