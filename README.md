# Toaster Chess Analysis

PGN ingestion and Stockfish analysis for games exported from Android chess apps.

## Flow

Android chess app exports PGN.
KDE Connect sends PGN to desktop.
Script ingests PGNs from `~/Downloads/kde_connect`.
Stockfish generates Markdown reports.
Reports are committed and pushed to GitHub for phone-friendly review.
