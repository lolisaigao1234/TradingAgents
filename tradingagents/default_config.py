import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": os.getenv("TRADINGAGENTS_LLM_PROVIDER", "google"),
    "deep_think_llm": os.getenv("TRADINGAGENTS_DEEP_MODEL", "gemini-3.1-pro-preview"),
    "quick_think_llm": os.getenv("TRADINGAGENTS_QUICK_MODEL", "gemini-3.1-flash-lite-preview"),
    "backend_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    # Provider-specific thinking configuration
    "google_thinking_level": os.getenv("TRADINGAGENTS_GOOGLE_THINKING_LEVEL"),      # "high", "minimal", etc.
    "google_api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    "google_vertexai": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true",
    "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT", "acacia-73dce"),
    "google_cloud_location": os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    "google_service_account_json": os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
    "google_service_account_secret_id": os.getenv("GOOGLE_VERTEX_SERVICE_ACCOUNT_SECRET_ID", "vertex-embed/oauth/serviceAccountJson"),
    "openai_reasoning_effort": os.getenv("TRADINGAGENTS_OPENAI_REASONING_EFFORT"),    # "medium", "high", "low"
    # LLM request settings
    "llm_timeout": 120,       # per-request timeout in seconds
    "llm_max_retries": 1,     # max retries on transient failures
    # Evaluator settings
    "enable_evaluator": False,
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, coingecko (auto for crypto)
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance, coingecko (auto for crypto)
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance, coingecko (auto for crypto)
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance; wsj always supplements
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # CoinGecko configuration (optional — free tier works without key)
    "coingecko_api_key": os.getenv("COINGECKO_API_KEY"),
    # WSJ news integration
    "wsj_scraper_path": "/mnt/acacia_rw/scripts/wsj-scraper.py",
    "wsj_max_scrape_articles": 3,
}
