"""
Utility functions for URL management and watchlist.
"""
import streamlit as st


def get_watchlist_from_url():
    """Get watchlist from URL query parameters."""
    # Support both old and new Streamlit query params API
    try:
        # New API (Streamlit >= 1.28.0)
        tickers = st.query_params.get("tickers", [""])
    except AttributeError:
        # Old API (Streamlit < 1.28.0)
        try:
            query_params = st.experimental_get_query_params()
            tickers = query_params.get("tickers", [""])
        except AttributeError:
            # Fallback if neither API is available
            return []
    
    # Handle both list and string formats
    if isinstance(tickers, list) and len(tickers) > 0:
        ticker_str = tickers[0] if tickers[0] else ""
    elif isinstance(tickers, str):
        ticker_str = tickers
    else:
        ticker_str = ""
    
    return [t.strip().upper() for t in ticker_str.split(",") if t.strip()] if ticker_str else []


def set_watchlist_to_url(tickers):
    """Save watchlist to URL query parameters."""
    # Support both old and new Streamlit query params API
    try:
        # New API (Streamlit >= 1.28.0)
        if tickers:
            st.query_params["tickers"] = ",".join(tickers)
        else:
            # Clear the tickers parameter
            if "tickers" in st.query_params:
                del st.query_params["tickers"]
    except AttributeError:
        # Old API (Streamlit < 1.28.0)
        try:
            if tickers:
                st.experimental_set_query_params(tickers=",".join(tickers))
            else:
                st.experimental_set_query_params()  # Clear parameters
        except AttributeError:
            # If neither API is available, silently fail
            pass
