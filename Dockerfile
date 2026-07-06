# syntax=docker/dockerfile:1
# starglyph-serve container (Stage 0 · C3). Build context = repo root:
#   docker build -t starglyph-serve .
#   docker run -d -p 127.0.0.1:8080:8080 -v starglyph-data:/var/lib/starglyph \
#     --cpus 2 --memory 2g starglyph-serve
# The image ships the committed HYG catalog and constellation data; the
# volume holds the pattern-database cache (and telemetry), so databases are
# generated on the first ever start and survive restarts. See docs/serve.md.

FROM rust:1.96-slim-bookworm AS build
WORKDIR /src
COPY prototype/ prototype/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/prototype/target \
    cargo build --release --manifest-path prototype/Cargo.toml -p starglyph-serve \
    && cp prototype/target/release/starglyph-serve /usr/local/bin/starglyph-serve

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --user-group starglyph \
    && mkdir -p /var/lib/starglyph \
    && chown starglyph:starglyph /var/lib/starglyph

COPY --from=build /usr/local/bin/starglyph-serve /usr/local/bin/starglyph-serve
COPY data/catalogs/hyg_v42.csv.gz /opt/starglyph/data/catalogs/
COPY data/celestial/constellations.lines.json \
     data/celestial/constellations.json \
     /opt/starglyph/data/celestial/

# Compile-time repo-relative defaults do not exist inside the image, so every
# path is pinned via env (flags still override; see `starglyph-serve --help`).
ENV STARGLYPH_SERVE_ADDR=0.0.0.0:8080 \
    STARGLYPH_SERVE_CATALOG=/opt/starglyph/data/catalogs/hyg_v42.csv.gz \
    STARGLYPH_SERVE_LINES=/opt/starglyph/data/celestial/constellations.lines.json \
    STARGLYPH_SERVE_NAMES=/opt/starglyph/data/celestial/constellations.json \
    STARGLYPH_SERVE_CACHE_DIR=/var/lib/starglyph/cache \
    STARGLYPH_SERVE_TELEMETRY_LOG=/var/lib/starglyph/telemetry/solve-log.jsonl

USER starglyph
VOLUME /var/lib/starglyph
EXPOSE 8080
# Liveness only: the listener is up from the start; readiness is GET /readyz.
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1
ENTRYPOINT ["/usr/local/bin/starglyph-serve"]
