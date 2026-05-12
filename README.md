# Toaster Chess Analysis

PGN ingestion and Stockfish analysis for games exported from my crappy android chess app.

## Flow

it export PGN.
KDE Connect sends PGN to desktop.
Script ingests PGNs from `~/Downloads/kde_connect`.
Stockfish generates Markdown reports.
ollama on my desktop takes what stockfish says and makes a sentence that sounds neat
Reports are committed and pushed to GitHub for phone-friendly review.
