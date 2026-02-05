"""
Utility functions for URL management and watchlist.
"""
import streamlit as st


def get_watchlist_from_url():
    """Get watchlist from URL query parameters."""
    tickers = st.query_params.get("tickers", [""])
    return [t.strip().upper() for t in tickers[0].split(",") if t.strip()] if tickers[0] else []


def set_watchlist_to_url(tickers):
    """Save watchlist to URL query parameters."""
    if tickers:
        st.query_params["tickers"] = ",".join(tickers)
    else:
        # Clear the tickers parameter
        if "tickers" in st.query_params:
            del st.query_params["tickers"]
