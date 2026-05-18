# Task 1, Fail Taksonomisi (LongMemEval-S, run5_v335_hybrid_v4_ft300)

**Tarih:** 2026-05-16
**Run:** `benchmarks/v335/run5_v335_hybrid_v4_ft300.jsonl` (hybrid_v4 + FT-300)
**Toplam fail:** 25/500 (R@1 = 0.95)
**Bu analizde:** 9 preference + 6 temporal = 15 fail (en yüksek puan getiri potansiyeli)

## Kategori tanımları

- **(a) annotation noise**, eval yanlış etiketlemiş; model aslında doğruyu buldu, ama gold farklı session işaretliyor
- **(b) multi-correct**, birden fazla session doğru; gold yalnızca birini istiyor, model başka bir doğruyu getirdi
- **(c) gerçek model hatası**, doğru session korpus'ta var, model semantik olarak uzaklaştı (ya top-1 alakasız ya retrieval hiç bulamadı)

---

## Preference fails (9 toplam)

| # | qid | top-1 ile gold uzaklığı | Doğru session top-K içinde? | Kategori |
|---|---|---|---|---|
| 1 | 0edc2aef (Miami hotel / "ocean+skyline view") | top-1 = Las Vegas paketleme (alakasız). top-2 = Seattle hotel "great view of city", gold preference'ı (view tercihi) örneklemiyor; gold session farklı | top-5 içinde "view" sinyali var ama Miami session değil | (c) ranking + retrieval miss |
| 2 | 06f04340 (homegrown dinner / "cherry tomatoes, basil, mint") | top-1 = mixed greens lunch. top-5 = "fresh basil and mint recipes, basil mint tomatoes", **gold session bu** | ✅ #5 doğru | (c) ranking error |
| 3 | 38146c39 (cookies / "turbinado sugar") | top-1 = cherry clafoutis. top-2 = "experimenting with turbinado sugar", **gold session** | ✅ #2 doğru | (c) ranking error |
| 4 | d24813b1 (bake for colleagues / "lemon poppyseed cake") | top-1..5 = slow cooker / chocolate cake / popcorn / vegan pasta / chicken, lemon poppyseed yok | ❌ top-5'te yok | (c) retrieval miss |
| 5 | 57f827a0 (rearrange bedroom / "mid-century modern dresser") | top-1 = plumber. top-3 = "mid-century modern bedroom dresser", **tam gold** | ✅ #3 doğru | (c) ranking error |
| 6 | 95228167 (new guitar / "Fender Strat vs Gibson Les Paul") | top-1 = Levi's jeans. top-5 = "Fender Stratocaster vs Gibson Les Paul", **tam gold** | ✅ #5 doğru | (c) ranking error |
| 7 | 505af2f5 (coffee creamer / "almond milk + vanilla + honey") | top-1 = Target shopping. top-2 = "almond milk, vanilla extract, honey creamer", **tam gold** | ✅ #2 doğru | (c) ranking error |
| 8 | d6233ab6 (high school reunion / "positive HS experiences") | top-1 = Khalid concert. top-4 = "happy high school memories", gold sinyali | ✅ #4 kısmi doğru | (c) ranking error |
| 9 | 1c0ddc50 (commute activities / "podcasts beyond true crime/self-improvement") | top-1 = bike commute logistics ("commute" lexical match, preference yok). top-2 = "podcasts true crime self-improvement commute", **tam gold** | ✅ #2 doğru | (c) ranking error |

**Preference özeti:**
- (a) annotation noise: **0/9**
- (b) multi-correct: **0/9**
- (c) gerçek model hatası: **9/9**, bunun **8/9'unda gold cevap top-5 içinde**, sadece 1 tanesinde (#4 lemon poppyseed) retrieval tamamen kaçırmış
- **8 fail rerank ile kurtulabilir**, cross-encoder ya da preference-tuned head top-5'i yeniden sıralarsa
- Pattern: hybrid_v4 (BM25 + dense) lexical "commute / bedroom / cake / cookies" sinyaline yakalanıyor, preference cümlesindeki implicit semantiği (turbinado, mid-century, Fender, almond+vanilla+honey) yeterince ödüllendirmiyor

## Temporal fails (6 toplam)

| # | qid | Soru | Gold | Top-5'te gold var mı? | Kategori |
|---|---|---|---|---|---|
| 1 | gpt4_59149c78 | "art event two weeks ago" | Metropolitan Museum of Art | ❌ top-5 = MoMA, modern art tour, mummification, crafting, Met yok | (c) retrieval miss (yakın MoMA confusion) |
| 2 | gpt4_4929293b | "relative's life event one week ago" | cousin's wedding | ❌ top-5 = baby gift, chapter presentation, own wedding planning, cousin yok | (c) retrieval miss |
| 3 | gpt4_468eb064 | "lunch last Tuesday, who?" | Emma | ❌ top-5 Tuesday-dated ama Emma adı yok | (c) retrieval miss |
| 4 | eac54add | "business milestone four weeks ago" | "signed contract with first client" | ❌ top-5 = flossing, to-do list, reading list, 5K running, Ibadan industries | (c) retrieval miss |
| 5 | 4dfccbf8 | "what did I do with Rachel Wednesday two months ago" | "started ukulele lessons with Rachel" | ❌ top-5'te Rachel adı bile yok | (c) retrieval miss (en kötü, entity match bile yok) |
| 6 | gpt4_93159ced_abs | "how long working before Google" | **abstain**: "haven't started Google yet" |, | (a/özel) **abstain question**, model herhangi bir doc getirse fail; eval-protokol farkı, retrieval düzeltmesiyle kapanmaz |

**Temporal özeti:**
- (a) annotation noise / abstain: **1/6** (#6), yapısal, ranking ile kurtarılamaz
- (b) multi-correct: **0/6**
- (c) gerçek retrieval miss: **5/6**, **hiçbirinde gold cevap top-5 içinde değil**, cross-encoder rerank fayda etmez, retrieval pay'ını (zaman filter, lexical entity boost) düzeltmek gerek
- Pattern: relatif zaman ifadesi ("two weeks ago", "last Tuesday", "four weeks ago") embedding'de güçlü sinyal değil; hybrid_v4 zaman fonksiyonu ya yok ya zayıf

---

## Tavan tahmini

**Annotation noise + abstain edge case:** 1/15 fail (Temporal #6). Tüm 500 üzerinden 1/500 = 0.002 yapısal tavan kaybı sadece bu kategorilerden. Diğer 24 fail (non-preference, non-temporal) ayrı incelenmedi ama benzer oranda noise olursa toplam ~3-5 fail noise; **noise-adjusted tavan ≈ R@1 0.99-0.994** (yani 0.99 _ulaşılabilir_).

**Pratik tavan tahmini, mevcut yaklaşımlarla:**

| Müdahale | Beklenen recovery | Yorum |
|---|---|---|
| Preference cross-encoder rerank top-20 → top-1 | 6-8 / 9 preference fail | 8'inde gold top-5'te; rerank doğrudan iş yapar |
| Preference hard-neg + sentetik FT | ek 1-2 fail (Task 4 sonrası kalan) | Task 4 ile büyük ölçüde örtüşür; marjinal |
| Temporal time-aware rerank (timestamp filter + lexical entity boost) | 2-4 / 5 temporal fail | "Rachel" "Metropolitan" "cousin wedding" gibi entity match'i artırır; "Tuesday Emma" gibilerini retrieval'da yakalamak gerek |
| Temporal #6 (abstain) | 0 | Yapısal, eval protokol değişikliği gerek |

**Realist hedef (preference + temporal odaklı):**
- 9 preference fail → 1-2'ye iner (Task 4 + Task 2)
- 6 temporal fail → 2-3'e iner (Task 3); #6 kalır
- Diğer 10 fail (multi-session 5, knowledge 1, single-user 2, single-assistant 2): bu sprint'te dokunulmuyor; aynı 10 kalır

**Beklenen final R@1:** 0.95 → **0.965-0.975** (sprint sonu, gerçekçi).
**0.99 için:** kalan 10 fail kategorisi de işlenmeli + noise temizliği yapılmalı; bu sprint'in scope'u dışında, 2-3 günde ulaşılmaz.

**Verdict:** Atakan'ın 2-3 günlük tahmini ile uyumlu; 0.99 hedefi ulaşılabilir görünüyor ama bu sprint sadece preference+temporal'a odaklanırsa tavanı 0.975 civarı. 0.99 için ikinci sprint (multi-session + diğer kategoriler) gerek. Noise oranı çok düşük (1/15), yani anchor budur, atılan her doğru fail kazanca dönüyor.
