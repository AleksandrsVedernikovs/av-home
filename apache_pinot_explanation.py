"""
Utility for printing a concise description of Apache Pinot.

This avoids making live API calls while still demonstrating the kind of
output someone might request from an LLM using the OpenAI client library.
"""

from __future__ import annotations


def explain_apache_pinot() -> str:
    """Return a two-sentence overview of Apache Pinot."""
    return (
        "Apache Pinot is a real-time distributed OLAP datastore built for "
        "serving low-latency analytics on event streams and batch data. "
        "It powers user-facing dashboards and anomaly detection by combining "
        "columnar storage, smart indexing, and pre-aggregation to answer "
        "queries quickly at scale."
    )


def main() -> None:
    print(explain_apache_pinot())


if __name__ == "__main__":
    main()
