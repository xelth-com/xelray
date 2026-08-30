# locales/

`en.json` is a generated manifest of every translatable string in XelRay,
extracted from the `EN` table in
[`crates/app_ui/src/i18n.rs`](../crates/app_ui/src/i18n.rs).

**It is documentation, not a runtime asset.** The English strings are
compiled into the WebAssembly binary, so the app is complete in English with
no files and no network. Nothing here is fetched or shipped in `dist/`.

It exists for two reasons:

- A translator can see the whole key set and its source text in one place.
- Whoever maintains a `/i18n/{lang}` backend can import these keys into it.

## How translations reach the app

At runtime XelRay asks its own origin for `/i18n/{lang}` — the same endpoint
[xelth.com](https://xelth.com) serves — and merges whatever comes back over
the embedded English. Any key the backend does not supply keeps its embedded
value, so a partial translation degrades key by key rather than all at once.

If the fetch 404s, times out, or there is no server at all (self-hosted,
`file://`, a USB stick in a hospital corridor), nothing happens and the app
stays English. First render never waits on it.

## Regenerating

The `EN` table in `i18n.rs` is the single source of truth; this file is
derived from it. Regenerate after adding or changing a string.
