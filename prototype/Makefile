fetch-catalog:
	bash ../scripts/fetch-hyg-catalog.sh

validate-simulator:
	cargo fmt --check
	cargo test -p simulator-core

validate-solver:
	cargo fmt --check
	cargo test -p solver-core

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
