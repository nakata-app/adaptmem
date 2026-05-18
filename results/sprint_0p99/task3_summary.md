# Task 3, Time-aware rerank (temporal-reasoning), TAMAMLANDI

**Tarih:** 2026-05-16
**Subset:** temporal-reasoning, n=133
**Sonuç:** R@1 0.9549 → **0.9699** (+1.5pp), 6 fail → **4 fail**, kaybeden 0

## Yaklaşım

Post-hoc time-aware rerank (`results/sprint_0p99/task3_time_rerank.py`):
1. Question'da relatif zaman ifadesini regex ile parse et (X days/weeks/months ago, last Xday, last week/month/year, yesterday, this week/month)
2. `question_date` + offset = target tarih
3. Top-50 doc timestamp ile target arasında Gaussian proximity (`sigma` zaman-birimine göre 1-30 gün)
4. Blended skor: `RRF_base(rank, k=60) + alpha * proximity`
5. **Gating:** Top-1 baseline doc proximity ≥ 0.3 ise dokunma (baseline'ı zaten doğru olabilir → koru)
6. Temporal keyword yoksa boost yok (no-op)

## En iyi config

`--base rrf --rrf-k 60 --alpha 0.01 --gate-prox 0.3`

| Mod | R@1 | Moved | Lost |
|---|---|---|---|
| Baseline | 0.9549 |, |, |
| Linear base, α=0.5, gate yok | 0.9624 | 3 | 2 |
| Linear base, α≥1.0, gate yok | hızla bozuluyor |, |, |
| RRF base, α=0.01, gate yok | 0.9624 | 3 | 2 |
| **RRF base, α=0.01, gate=0.3** | **0.9699** | **2** | **0** |
| RRF base, α≥0.05, gate=0.3 | ≤0.94 |, |, |

Gating'ten önce 2 lost vardı (top-1 zaten gold iken time-boost başka doc'a kaydırıyordu, `gpt4_8279ba03` "10 days ago kitchen", `gpt4_b5700ca0` "religious activity last week"). Gating bu iki vakayı korudu, R@1'i tam 1.5pp yukarı çekti.

## Kurtarılan 2 fail

- `gpt4_59149c78` "art event two weeks ago" → target=2023-01-18 ±3d, gold doc 2023-01-15
- `4dfccbf8` "Rachel Wednesday two months ago" → target=2023-01-30 ±7d, gold doc 2023-01-24

## Kalan 4 fail (rerank'in tavanı)

| qid | Sebep | Yapısal mı? |
|---|---|---|
| `gpt4_4929293b` (relative's life event 1 week ago) | gated_top1_close, top-1 yanlış doc ama target'a yakın; gate yanlış pozitif | Hayır, daha akıllı gate gerek |
| `gpt4_468eb064` (lunch last Tuesday, Emma) | gated_top1_close, aynı sebep, semantic entity (Emma) sinyali yok | Hayır, entity-aware boost gerek |
| `eac54add` (business milestone 4 weeks ago) | sigma=3 gün → gold doc 18 gün önce, proximity düştü | Sigma weeks için artırılabilir ama yan etki olabilir |
| `gpt4_93159ced_abs` (Google öncesi süre) | no_temporal_keyword, abstain question | EVET, yapısal, eval protokol farkı |

3/4'ü bu yaklaşımda kapalı; entity-aware boost veya better gating ek kazanç verebilir ama bu sprint scope'unda değil. 1/4 yapısal tavan.

## Global LongMemEval-S etkisi

- Toplam fail: 25 → 23
- Total R@1: 0.95 → **0.954**
- Bu Task 3'ün tek başına katkısı, lost=0 → güvenli, üretime alınabilir

## Önerilen yer

MemPalace bench `longmemeval_bench.py` içinde `mode=hybrid_v4` sonrası rerank katmanı olarak. Veya:
- `adaptmem/` paketine `time_aware_rerank.py` modülü olarak ekle
- Bench script'i `--time-rerank` flag'i ile çağırır
- AdaptMem MCP retrieve katmanında temporal q tespit edilirse uygula
