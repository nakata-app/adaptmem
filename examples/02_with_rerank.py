"""Cross-encoder rerank example.

The bi-encoder is fast but noisy at the top-1 position. The CE rerank stage
reorders a wider candidate set with a slower but stronger model. Useful when
R@1 matters more than throughput.

  python examples/02_with_rerank.py
"""
from __future__ import annotations

from adaptmem import AdaptMem


def main():
    corpus = [
        {"id": "a1", "text": "Apache Kafka is a distributed event streaming platform."},
        {"id": "a2", "text": "Kafka topics are partitioned for parallelism and replication."},
        {"id": "a3", "text": "RabbitMQ is a traditional message broker built on AMQP."},
        {"id": "a4", "text": "NATS is a lightweight publish-subscribe messaging system."},
        {"id": "a5", "text": "Pulsar is a cloud-native distributed messaging platform."},
        {"id": "a6", "text": "Kafka consumers join groups so each partition is read by one member at a time."},
    ]
    labelled = [
        {"query": "kafka partition consumer model", "relevant_ids": ["a2", "a6"]},
        {"query": "AMQP message broker",            "relevant_ids": ["a3"]},
        {"query": "lightweight pub/sub system",     "relevant_ids": ["a4"]},
    ]

    # rerank=True persists the choice into the saved model dir, so .load()
    # restores it. CE weights aren't downloaded until the first .search()
    # call — fine for tests.
    am = AdaptMem(
        base_model="all-MiniLM-L6-v2",
        rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        device="cpu",
    )
    am.train(corpus=corpus, labelled=labelled)

    # rerank_top_k controls how wide the bi-encoder candidate set is before
    # the CE reorders it. Default = top_k * 3.
    hits = am.search("kafka partition rebalance", top_k=3, rerank_top_k=6)
    print("CE-reranked top-3:")
    for h in hits:
        print(f"  {h.score:+.3f}  {h.chunk_id}  {h.text[:70]}")
    print()
    print("Note: hit.score is the CE logit when rerank is on, not a cosine.")


if __name__ == "__main__":
    main()
