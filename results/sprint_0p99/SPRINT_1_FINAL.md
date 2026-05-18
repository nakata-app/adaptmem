# Sprint 1, AdaptMem LongMemEval-S 0.95 → 0.99 (KAPANIŞ)

**Tarih:** 2026-05-16
**Kapsam:** preference + temporal-reasoning fails

## Net etki

| Metrik | Sprint öncesi | Sprint sonu | Δ |
|---|---|---|---|
| LongMemEval-S R@1 | 0.95 | **0.954** | +0.4pp |
| Fails (toplam 500) | 25 | 23 | -2 |
| Temporal R@1 | 0.9549 | **0.9699** | +1.5pp |
| Temporal fails | 6 | 4 | -2 |
| Preference R@1 | 0.70 | 0.70 | 0 |

## Üretime alınabilir çıktı

**Time-aware rerank (Task 3):** `results/sprint_0p99/task3_time_rerank.py`
- Config: `--base rrf --rrf-k 60 --alpha 0.01 --gate-prox 0.3`
- Post-hoc, MemPalace pipeline'a dokunmadan
- lost=0 → güvenli, prod'a alınabilir
- Kuracağı yer: MemPalace `longmemeval_bench.py` `mode=hybrid_v4` sonrası, ya da AdaptMem MCP retrieve katmanı

## Kapsam dışı kalan (Sprint 2'ye carryover)

**Task 2, preference FT (PARTIAL):**
- `chat-ce-v1-20260516` ckpt eğitildi (Kaggle T4, ms-marco-MiniLM-L-12-v2 + 848 hard-neg pair, 3 epoch, val acc 0.9484)
- Kaydedildi: `checkpoints/chat-ce-v1-20260516/`
- Train data: 100 train split'inden, preference n=7 yetersiz
- Sprint 2'de: DeepSeek V4 Pro (NIM) ile preference paraphrase augmentation, retrain → v2

**Task 4, cross-encoder rerank (NET 0):**
- codecrossenc-v2 (kod modeli): -79pp (kapı dışı)
- ms-marco zero-shot: -3pp
- chat-ce-v1 pure: -3.7pp
- chat-ce-v1 RRF k=60: -1.2pp
- chat-ce-v1 preference-only: net 0 (4 kazandı 4 kaybetti)
- Sprint 2'de: chat-ce-v2 ile re-run

## Önemli bulgu (Task 1 sırasında keşfedildi)

`fails_unrecoverable_topk = 0`, preference+temporal fails'lerin **TÜMÜNDE gold cevap top-20 içinde mevcut**. Yani retrieval miss değil, sadece ranking hatası → doğru reranker bulunursa 15/15 fail kurtarılabilir.

## Sprint 2 önerilen plan

1. **Task A**, DeepSeek V4 Pro ile preference augmentation (7 → 200-500 q), chat-ce-v2 retrain (~2 sa toplam)
2. **Task B**, multi-session fail analizi (5 fail, dokunulmadı)
3. **Task C**, knowledge-update + single-session fails (5 fail)
4. **Task D**, annotation noise temizliği (1 abstain + olası noise'lar, noise-adjusted tavan ~0.992)

**Realist hedef:** Sprint 2 sonu R@1 **0.97-0.98**. 0.99 Sprint 3 işi.

## Dosyalar

- `task1_fail_taxonomy.md`, 15 fail taksonomi raporu
- `task3_time_rerank.py` + `task3_summary.md`, üretime alınabilir
- `task3_final.json`, final temporal sonuç (R@1 0.9699)
- `task4_summary.md`, cross-encoder rerank denemeleri
- `task4_chat_pure.json`, chat-ce-v1 final eval
- `task2_train_pairs.jsonl` (848) + `task2_val_pairs.jsonl` (155)
- `task2_train.py` (Mac, kullanılmadı, OOM) + `kaggle_bundle/train_chat_ce.ipynb`
- `checkpoints/chat-ce-v1-20260516/`, model weights (127 MB)
