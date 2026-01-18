"""
Utility functions for URL management and watchlist.
"""
import streamlit as st


def get_watchlist_from_url():
    """Get watchlist from URL query parameters."""
    query_params = st.experimental_get_query_params()
    tickers = query_params.get("tickers", [""])
    return [t.strip().upper() for t in tickers[0].split(",") if t.strip()] if tickers[0] else []


def set_watchlist_to_url(tickers):
    """Save watchlist to URL query parameters."""
    if tickers:
        st.experimental_set_query_params(tickers=",".join(tickers))
    else:
        st.experimental_set_query_params()  # Clear parameters
