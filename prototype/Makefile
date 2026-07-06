# "eval" collides with the checked-in eval/ directory (baseline-ci.json);
# without .PHONY make considers the target permanently up to date.
.PHONY: fetch-catalog validate-simulator validate-solver validate-serve eval eval-gate serve-smoke

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
