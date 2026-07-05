# Contributing to Starglyph

Thanks for your interest in contributing!

## License of contributions

Starglyph's engine, CLI, and desktop app are licensed under the
**Apache License 2.0** (see [LICENSE](LICENSE)). By contributing, you agree that
your contributions are licensed under Apache-2.0 as well (inbound = outbound).

Bundled datasets are under their own licenses (HYG: CC BY-SA 4.0; d3-celestial:
BSD-3-Clause) — see [ATTRIBUTION.md](ATTRIBUTION.md). Do not add data or code
under GPL, LGPL, AGPL, SSPL, or other copyleft licenses; the CI license check
(`cargo deny check licenses`, config in [deny.toml](deny.toml)) will reject them.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/)
to certify the provenance of contributions. It is lightweight: no separate CLA
to sign. You certify the DCO by adding a `Signed-off-by` line to each commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

Add it automatically with:

```bash
git commit -s
```

The name and email must be real and match the commit author. By signing off you
assert the DCO: that you wrote the change or have the right to submit it under
the project's Apache-2.0 license.

> Note: the DCO keeps the project inbound = outbound (Apache-2.0). It does **not**
> transfer copyright. If the project later needs to offer the engine under a
> separate commercial license, that would require a CLA instead — which is why
> the current choice is Apache-2.0 (permissive), where a CLA is not needed to
> ship the closed web/mobile products.

## Development

See [docs/](docs/README.md) for architecture, roadmap, and data contracts.
Before opening a PR:

- `cargo test` passes,
- `cargo clippy` is clean,
- `cargo fmt` applied,
- new third-party dependencies pass `cargo deny check licenses`.
