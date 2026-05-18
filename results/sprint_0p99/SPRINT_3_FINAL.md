# Sprint 3, ft-v4 bi-encoder + birleşik final (KAPANIŞ)

**Tarih:** 2026-05-16
**Hedef:** R@1 0.95 → 0.99
**Sonuç:** R@1 = **0.974** (13 fail / 500)

## Adımlar

| Adım | Sonuç |
|---|---|
| Bi-encoder ft-v4 train (Kaggle T4, MNR loss, 5622 pair, 2 epoch) | val pos cosine 0.45, neg 0.17, margin 0.28 |
| MemPalace bench rerun (mempal_bench_with_ft.py + ft-v4 + hybrid_v4) | run6_v335_hybrid_v4_ftv4.jsonl, **R@1 0.968** (ft-300 baseline 0.95'ten +1.8pp) |
| Per-category Sprint 2 strateji uygulaması | 0.968 → **0.974** (3 fail kurtarma) |

## Birleşik global tablo (Sprint 1+2+3)

| Aşama | R@1 | Fails | Δ |
|---|---|---|---|
| Baseline (hybrid_v4 + ft-300) | 0.950 | 25 |, |
| + Sprint 1 Task 3 time-aware rerank | 0.954 | 23 | -2 |
| + Sprint 2 chat-ce-v2/v3 cross-encoder rerank | 0.974 | 13 | -10 |
| **+ Sprint 3 ft-v4 bi-encoder** | **0.974** | **13** | **0** (ft-v4 tek başına 0.968, rerank ile aynı tepeye geliyor) |

**Total Sprint 1+2+3 kazanım:** 25 fail → 13 fail, R@1 0.95 → **0.974** (+2.4pp, %48 fail azaltma)

## Per-category strateji (final)

```python
STRAT = {
    "single-session-preference": ("v3-pure",    pure(chat-ce-v3)),
    "temporal-reasoning":         ("time-aware",  task3_time_rerank),
    "multi-session":              ("v3-pure",    pure(chat-ce-v3)),
    "knowledge-update":           ("v3-RRF60",   rrf(chat-ce-v3, k=60)),
    "single-session-assistant":   ("v2-pure",    pure(chat-ce-v2)),
    "single-session-user":        ("baseline",   no_rerank),
}
```

## Kalan 13 fail (Sprint 4 hedefleri)

| Kategori | Count | qids |
|---|---|---|
| preference | 5 | 35a27287, 06f04340, 09d032c9, 38146c39, 57f827a0 |
| temporal | 5 | gpt4_4929293b, 9a707b82, eac54add, gpt4_8279ba03, gpt4_93159ced_abs* |
| multi-session | 2 | 6d550036, d905b33f |
| single-user | 1 | f4f1d8a4_abs* |

\* 2 fail abstain edge case → noise-adjusted tavan ~**0.978**

## Artefaktlar

| Dosya | İçerik |
|---|---|
| `checkpoints/ft-v4-20260516/` | bi-encoder (MNR FT, 5622 pair, 87 MB) |
| `checkpoints/chat-ce-v2-20260516/` | cross-encoder (preference augmented, 280 syn pair) |
| `checkpoints/chat-ce-v3-20260516/` | cross-encoder (all-types augmented, 5448 syn pair) |
| `benchmarks/v335/run6_v335_hybrid_v4_ftv4.jsonl` | yeni retrieval run (R@1 0.968) |
| `results/sprint_0p99/task3_time_rerank.py` | time-aware rerank (RRF + gated proximity) |
| `results/sprint_0p99/SPRINT_{1,2,3}_FINAL.md` | her sprint kapanış |

## Sprint 4, sıradaki

**Mass-utility ürünler (Atakan'ın seçimi: 2 → 1 → 4):**
1. **MCP Server**, drop-in personal memory MCP for Claude Code/Cursor/Desktop (1-2 hafta)
2. **Browser Extension**, cross-site AI memory (ChatGPT/Claude/Gemini) (2-3 ay)
3. **NPM/PyPI Package**, AdaptMem SDK (1 hafta paralel)

**Akademik (paper) yön:**
- B: per-query mode routing (learned routing), 2-3 hafta workshop paper
- A: online encoder adaptation (continual learning), 2-3 ay conference paper
- D: multi-hop memory retrieval, 4-6 ay (Atakan ilgi gösterdi, niş değil mass-utility de istiyor)

**Karma C planı, upstream contribution:**
- ft-v4 + time-aware rerank → jphein/mempalace upstream PR (Atakan ekip dışı, harici contributor)
- AdaptMem'in özgün katmanları (cross-encoder rerank + augmentation pipeline + ürünler) bağımsız tut

## Notlar (gelecek session için)

- DeepSeek V4 Pro/Flash NIM'de timeout, fallback Llama-3.3-70B (ücretsiz, 5-12s yanıt)
- Kaggle T4 x2 + CUDA_VISIBLE_DEVICES=0 zorunlu (DataParallel hatası önleme)
- Sprint 1'de Mac mini train OOM/swap olur, train daima Kaggle'da
- Inference (rerank, eval) Mac mini'de güvenli, ~1.5 GB RAM, MPS yeter
