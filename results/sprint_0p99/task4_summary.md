# Task 4, Cross-encoder rerank (subset), SONUÇ: NET NEGATİF, Task 2'YE BAĞIMLI

**Tarih:** 2026-05-16
**Subset:** single-session-preference (30) + temporal-reasoning (133) = 163 q
**Baseline:** hybrid_v4 + FT-300, subset R@1 = 0.908 (preference 0.70, temporal 0.9549)

## Denemeler

| Reranker | Mod | Subset R@1 | Δ vs baseline | Verdict |
|---|---|---|---|---|
| codecrossenc-v2 (kod-domain) | pure | 0.117 | **-79.1pp** | NEGATIVE TRANSFER, kod domain, chat değil |
| ms-marco-MiniLM-L-12-v2 | pure | 0.8405 | -6.7pp | Net negatif (4 kazandı, 11 kaybetti) |
| ms-marco-MiniLM-L-12-v2 | RRF k=10 | 0.8405 | -6.7pp | Aynı |
| ms-marco-MiniLM-L-12-v2 | RRF k=20 | 0.8773 | -3.0pp | Hâlâ negatif |
| ms-marco-MiniLM-L-12-v2 | RRF k=40 | 0.8773 | -3.0pp | Aynı |
| ms-marco-MiniLM-L-12-v2 | RRF k=60 | 0.8773 | -3.0pp | Aynı, doyuyor |

## Niye çakıldı

1. **codecrossenc-v2** CodeSearchNet Python kod-doc çiftlerinde eğitilmiş. Sohbet preference ve temporal sorularında semantic match'i tamamen ters okuyor. Kullanılamaz.
2. **ms-marco zero-shot** generic web query→passage için. LongMemEval kullanıcı-asistan sohbet turnları MS MARCO passage'larına yapı olarak benzemiyor (uzun multi-turn, implicit referans). 4 doğru kurtardı ama 11 hatalı top-1 yaptı, net kayıp.
3. RRF füzyonu kaybı yumuşatıyor ama elimine etmiyor, bi-encoder zaten daha iyi karar veriyor, ce sinyal eklemiyor.

## Olumlu bulgu (Task 1 düzeltmesi)

`fails_unrecoverable_topk = 0`, **preference+temporal fails'lerin TÜMÜNDE gold cevap top-20 içinde mevcut**. Task 1'de manuel top-5 analizinde "retrieval miss" dediğim 5 temporal + 1 preference fail aslında top-6..20'de var. Yani:
- 15/15 fail teorik olarak rerank ile kurtarılabilir
- Eksiklik: doğru cross-encoder yok
- → Task 2 (chat-domain'de FT'lenmiş reranker) ön koşul haline geldi

## Sıradaki adım

Task 4'ü pause et. Task 2 (preference hard-neg + sentetik augmentation) tamamlandığında Task 4 yeni FT'li cross-encoder ile RE-RUN. O zaman beklenen: subset R@1 0.908 → 0.97+ (10-14 fail kurtarılır).

Bu arada Task 3 (time-aware rerank) bağımsız ilerleyebilir, temporal fails'in mantığı tarih/entity match, cross-encoder'a değil özel rule'a ihtiyaç var.
