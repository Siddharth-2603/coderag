# Caching

The retrieval layer uses a two-level cache. The first level is an in-memory LRU
cache that stores recent query results. The second level is a semantic cache
that matches new queries against previous ones by embedding similarity, so
paraphrased questions can reuse an earlier answer.

Cache entries expire after a configurable TTL. The semantic cache uses a
similarity threshold: only matches above the threshold are served from cache,
everything else falls through to a fresh retrieval.
