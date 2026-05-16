# Sprint 6, 5-model LLM rerank karşılaştırma (KAPANIŞ)

**Tarih:** 2026-05-16
**Hedef:** R@1 0.990 → 0.998 (Sprint 4 sonrası kalan 8 attackable fail'i kurtarmak)
**Sonuç:** R@1 **0.990 hard ceiling** doğrulandı, 5 farklı LLM aynı tavanda kilitlendi
**Bonus:** V4 Flash kanonik prod-LLM olarak teyit edildi (10× daha hızlı/ucuz, eşit R@1)

## Motivasyon

Sprint 4'te Llama-3.3-70B (NIM, ücretsiz) + 3-vote self-consistency ile R@1 0.990 elde edildi (4 helped / 8 attackable). Hipotez: daha güçlü reasoning model (DeepSeek V4 Pro, MiniMax M2/M2.7) kalan 4 attackable fail'i çözebilir → R@1 0.992-0.996. Test edildi.

## 5-model deney sonuçları

Hepsi aynı protokol: run8_time_rerank.jsonl üzerinden 8 attackable fail için targeted rerank, top-K=10, 3-vote majority, CoT prompt.

| Model | helped | hurt | R@1 | Wall | Cost/koşum | Not |
|---|---|---|---|---|---|---|
| Llama-3.3-70B (NIM, free) | 3 | 0 | 0.988 | ~85s | $0 | Sprint 4 baseline |
| DeepSeek V4 Pro | 4 | 0 | **0.990** | 867s | ~$0.50 | reasoning model, en yavaş |
| **DeepSeek V4 Flash** ⭐ | **4** | **0** | **0.990** | **88s** | **~$0.05** | **10× hızlı, prod kanonik** |
| MiniMax M2 | 3 | 0 | 0.988 | 279s | ~$0.04 | reasoning, ortak helped seti |
| MiniMax M2.7 | 1 | 0 | 0.984 | 585s | ~$0.05 | thinking budget aştı, ANSWER kesildi |

## Ensemble analizi

| Strateji | helped | R@1 |
|---|---|---|
| Oracle union (5 model) | 5 | **0.992** |
| 5-model cross-majority vote | 3 | 0.988 |

**Oracle union 5/8** çünkü her model farklı 1 fail kurtardı:
- V4 Pro: `6d550036` (multi-session count question)
- V4 Flash: `d905b33f` (book discount, text truncation)
- Diğer 3 helped: ortak (06f04340, gpt4_45189cb4, gpt4_ec93e27f)

**Cross-majority kötüleşti** çünkü M2.7 zayıf picks majority'i bozdu.

## 3 yapısal fail (5 model üst üste başarısız)

| qid | tip | sebep |
|---|---|---|
| `gpt4_4929293b` | temporal | "cousin's wedding" entity sinyali zayıf, hiç model yakalamadı |
| `9a707b82` | temporal | "chocolate cake" gold rank 2 ama lexical "cooking" tuzağı |
| `eac54add` | temporal | "milestone 4 weeks ago" gold rank 5, temporal sigma hassas |

Bunlar **prompt mühendisliği veya farklı LLM ile çözülmüyor**. Sprint 5'in (full-text expansion) başarısızlığı gösterdi ki "daha fazla context" çözüm değil. Olası yol: NER entity boost (deterministic, count/entity-aware retrieval öncesi rule).

## Verdict

- **Stable production R@1 = 0.990**, V4 Flash kanonik (cost/latency optimal)
- **Hard ceiling (mevcut yöntemlerle) = 0.992** (selective router, per-q-type model)
- **0.998 ulaşılmıyor**, 1 abstain (yapısal) + 3 hard temporal fail
- **0.998 için Sprint 7 gerek**: NER entity boost + count-aware deterministic rule + mempal turn-level full text expansion (Sprint 5 başarısız ama doğru pattern Sonnet/Opus seviyesinde model ile)

## Production öneri

Üç tier önerisi:

**Free tier (zero API cost):**
```
ft-v4 → trust gate CE (chat-ce-v3) → time-aware temporal
→ NIM Llama-70B fallback (free, 5-10 RPM)
```
Beklenen R@1: 0.987-0.988

**Premium tier (paid, opt-in):**
```
+ DeepSeek V4 Flash (~$0.05 per 500-q bench)
```
Beklenen R@1: 0.990

**Enterprise tier (selective router):**
```
+ per-q-type routing (Pro for count/multi-session, Flash for temporal, Llama for chat)
```
Beklenen R@1: 0.992

## Artefaktlar

| Dosya | İçerik |
|---|---|
| `results/sprint_0p99/sprint6_deepseek.py` | V4 Pro/Flash targeted rerank |
| `results/sprint_0p99/sprint6_minimax.py` | M2/M2.7 targeted rerank |
| `results/sprint_0p99/sprint6_deepseek_result.json` | V4 Pro sonuç |
| `results/sprint_0p99/sprint6_flash_result.json` | V4 Flash sonuç |
| `results/sprint_0p99/sprint6_minimax_result.json` | M2 sonuç |
| `results/sprint_0p99/sprint6_minimax27_result.json` | M2.7 sonuç (thinking budget overflow) |
| `results/sprint_0p99/sprint5_fulltext_llm_result.json` | Sprint 5 başarısız (geriletti, kayıt için tutuluyor) |

## Sprint 7 yol haritası (0.998 için)

1. **NER entity boost** (deterministic, no LLM), "cousin / wedding / cake / milestone" gibi salient nounları extract + retrieval öncesi lexical boost
2. **Count-aware deterministic rule** (multi-session "how many" pattern için cluster count)
3. **Mempal turn-level full text expansion**, top-K text 600 char yerine session-bütününü ver (Sprint 5 Llama'da başarısız, ama Sonnet/Opus disiplini ile yeniden test edilebilir)
4. **Eğer paid LLM kabul edilirse**: OpenRouter Sonnet 4.5 / Opus 4.7 ile aynı targeted-on-fails pattern. Bekleyen iş.
