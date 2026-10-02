# "eval" collides with the checked-in eval/ directory (baseline-ci.json);
# without .PHONY make considers the target permanently up to date.
.PHONY: fetch-catalog validate-simulator validate-solver validate-serve eval eval-gate eval-smartphone-gate eval-smartphone-sky-fill-gate test-automatic-sky serve-smoke docker-build docker-smoke

fetch-catalog:
	bash ../scripts/fetch-hyg-catalog.sh

validate-simulator:
	cargo fmt --check
	cargo test -p simulator-core

validate-solver:
	cargo fmt --check
	cargo test -p solver-core

validate-serve:
	cargo fmt --check
	cargo test -p starglyph-serve

# End-to-end: start the HTTP service, solve a real frame, hammer it in
# parallel, assert no pattern database is rebuilt. See scripts/serve-smoke.sh.
serve-smoke:
	bash ../scripts/serve-smoke.sh

docker-build:
	docker build -t starglyph-serve ..

# Container end-to-end: cold start on a fresh volume, solve, restart on the
# same volume with zero regeneration. See scripts/docker-smoke.sh.
docker-smoke:
	bash ../scripts/docker-smoke.sh

eval:
	cargo run --release -p starglyph-cli -- eval \
		--manifest ../data/samples/sky-samples/manifest.json \
		--out-dir artifacts/eval/local
	@echo "summary: artifacts/eval/local/summary.json"

eval-gate:
	cargo run --release -p starglyph-cli -- eval \
		--manifest ../data/samples/sky-samples/manifest.json \
		--ids tetra3_alt40,tetra3_alt60 \
		--catalog ../data/catalogs/hyg_v42.csv.gz \
		--out-dir artifacts/eval/ci \
		--baseline eval/baseline-ci.json

# Fresh run directories prevent stale solve reports from passing the gate.
eval-smartphone-gate:
	cargo build --release -p starglyph-cli --locked
	python3 eval/smartphone_gate.py

# Manual-mask experiment: protects 12 accepted IDs, not full-field astrometric truth.
eval-smartphone-sky-fill-gate:
	cargo build --release -p starglyph-cli --locked
	python3 eval/sky_fill_gate.py
	cargo test --release -p starglyph-core --test extended_dense -- --ignored

# Python-only research baseline; full three-arm runs require a fresh --out-dir.
test-automatic-sky:
	python3 -m unittest discover -s eval -p 'test_automatic_sky.py'
	python3 -m unittest discover -s eval -p 'test_gradient_sky.py'
	python3 -m unittest discover -s eval -p 'test_matching_search.py'
