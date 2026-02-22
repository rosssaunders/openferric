# OpenFerric Serialization Schemas

Canonical schema files for issue #52:

- `trade.schema.json`: tagged trade schema with product payload and trade metadata.
- `market_snapshot.schema.json`: point-in-time market data container (curves, surfaces, spot/forward).
- `pricing_audit.schema.json`: full pricing audit payload (`inputs -> model -> outputs`, including scenarios).

All payloads are designed for serde round-trip guarantees in both JSON (`serde_json`) and MessagePack (`rmp-serde`).
