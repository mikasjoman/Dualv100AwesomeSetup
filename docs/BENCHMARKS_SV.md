# Tesla V100 Dual Benchmark & Performance History

Denna tabell loggar och ackumulerar benchmark-körningar över tid för att utvärdera termik, energiförbrukning, tokens/sekund och fläkt/duct-effektivitet.

| Datum & Tid | Profil / Cap | Modell | Genomsnittlig Hastighet | GPU 0 Topptemp | GPU 1 Topptemp | Snitteffekt (GPU 0 / GPU 1) | Marginal till Nödstopp | Notering |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-10-25 Tidigare | 160W Cap | qwen3.8-flash-next-iq3_xxs | ~52.8 tok/s | 68.0°C | 74.0°C | ~140W totalt | +4.0°C (vid 78°C) | Ursprunglig duct, aggressiv fläkt |
| 2026-10-25 18:46 | 150W Cap | qwen3.8-flash-next-iq3_xxs | 59.3 tok/s | 59.0°C | 64.0°C | ~112W totalt | +14.0°C (vid 78°C) | Ny omdesignad tyst duct, 150W |
| 2026-10-25 18:49 | 155W Cap | qwen3.8-flash-next-iq3_xxs | 60.2 tok/s | 62.0°C | 68.0°C | 116.7W / 129.6W | +10.0°C (vid 78°C) | Ny duct, CPU balance_power, 3000 tokens |
| 2026-10-25 18:57 | 155W Cap, spec=3 | qwen3.8-flash-next-iq3_xxs | 62.9 tok/s (topp 68.1) | 61.0°C | 66.0°C | 114.2W / 130.2W | +12.0°C (vid 78°C) | MTP sänkt till spec=3, snabbare & svalare! |
| 2026-10-25 19:00 | 170W Cap, spec=3 | qwen3.8-flash-next-iq3_xxs | 63.6 tok/s (topp 68.8) | 61.0°C | 65.0°C | ~128W snitt | +13.0°C (vid 78°C) | Höjd effekt ger högre boost och stabilare decode |
| 2026-10-25 19:02 | 170W, per-layer cache | qwen3.8-flash-next-iq3_xxs | 55.0 tok/s (topp 56.2) | 60.0°C | 64.0°C | ~120W snitt | +14.0°C (vid 78°C) | Per-layer cache minskade träffytan för MoE; sämre |
| 2026-10-25 19:05 | 170W Cap, spec=3, opt | qwen3.8-flash-next-iq3_xxs | 60.6 tok/s (topp 65.7) | 60.0°C | 63.0°C | ~125W snitt | +15.0°C (vid 78°C) | Sparad som optimal standardprofil |
| 2026-10-25 19:08 | 170W Cap, spec=3, full 3k | qwen3.8-flash-next-iq3_xxs | 59.4 tok/s (topp 64.7) | 65.0°C | 71.0°C | 116.6W / 130.8W | +7.0°C (vid 78°C) | Fullt 5-rundors stresstest med alla optimeringar |
| 2026-10-26 21:18 | 150W/120W, Volta-Opt | qwen3.8-flash-next-ud-q4_k_xl | 43.1 tok/s (topp 45.7) | 59.0°C | 50.0°C | ~125W / ~75W | +18.0°C (vid 77°C) | Unsloth Q4 111GB, 40GB RAM-budget, STRATA_SM70_TABLE=1, STRATA_PF_WMMA=1 |
| 2026-10-26 22:30 | 150W/150W, NUMA1+1Cat | qwen3.8-flash-next-ud-q4_k_xl | 44.1 tok/s (topp 46.9) | 59.0°C | 52.0°C | ~130W / ~95W | +18.0°C (vid 77°C) | NUMA Node 1 Preferred, 1Cat Chunking (3072), spec=3 (p=0.38), Vision -> Strata |
| 2026-11-05 00:00 | 150W/150W, Strata v0.2.10 (0.1.56) | qwen3.8-flash-next-ud-q4_k_xl | **51.0 tok/s (topp 56.6)** | 69.0°C | 61.0°C | 122.4W / 81.9W | +8.0°C (vid 77°C) | Uppgraderad till v0.2.10 (`0.1.56`), `--adapt-async 1`, `--batch 2`, `--batch-mtp`, `STRATA_BATCH_MTP_SERIAL=1`, `STRATA_PW_WITH_BATCH=1` |
| 2026-11-05 00:01 | 150W/150W, 2x Parallella (Hermes) | qwen3.8-flash-next-ud-q4_k_xl | **61.9 tok/s totalt (topp 64.6, 31.1 tok/s per ström)** | 70.0°C | 61.0°C | ~130W / ~85W | +7.0°C (vid 77°C) | 2 parallella konversationer samtidigt (`--batch-mtp` 67.1% accept, 68.4 rows/s, direkt GPU->GPU slot-copy på ~209 ms) |


