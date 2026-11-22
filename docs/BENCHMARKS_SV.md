# Tesla V100 Dual Benchmark & Performance History

Denna tabell loggar den bästa verifierade benchmark-körningen per testdag (nyast överst) för att utvärdera termik, energiförbrukning, tokens/sekund och 120mm radialfläkt/duct-effektivitet.

| Datum | Profil / Cap | Modell | Bästa Hastighet för Dagen | GPU 0 Topptemp | GPU 1 Topptemp | Snitteffekt (GPU 0 / GPU 1) | Marginal till Nödstopp | Notering |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2026-11-22** | **140W/140W, Strata v0.2.19 (`0.1.65` `sm_70` + 120mm Radialfläkt @ 25%)** | **qwen3.8-flash-next-ud-q4_k_xl** | **93.6 tok/s parallell decode (90.0 E2E)** / **84.4 tok/s single decode (81.3 snitt)** | **48.0°C** | **49.0°C** | **114.5W / 93.9W** | **+28.0°C (vid 77°C)** | `SPLIT_CPU_SHARE=1` + `MMQ_TIER_SORT=1`, **0.18s varm TTFT**, **99.9–100% VRAM-träff**, **48–49°C topptemp vid 25% fläkt!** |
| **2026-11-05** | 150W/150W, Strata v0.2.10 (`0.1.56`, `--adapt-async 1 --batch-mtp`) | qwen3.8-flash-next-ud-q4_k_xl | **61.9 tok/s parallell (topp 64.6)** / **51.0 tok/s single (topp 56.6)** | 70.0°C | 61.0°C | 122.4W / 81.9W | +7.0°C (vid 77°C) | 2 parallella konversationer samtidigt (`--batch-mtp` 67.1% accept, 68.4 rows/s, direkt GPU->GPU slot-copy) |
| **2026-10-26** | 150W/150W, NUMA1 + Volta-Opt + 3072 Chunk | qwen3.8-flash-next-ud-q4_k_xl | **44.1 tok/s snitt (topp 46.9)** | 59.0°C | 52.0°C | ~130W / ~95W | +18.0°C (vid 77°C) | Unsloth Q4 111GB, NUMA Node 1 Preferred, 40GB RAM-budget, `STRATA_SM70_TABLE=1`, `STRATA_PF_WMMA=1` |
| **2026-10-25** | 170W Cap, `spec=3`, Ny Tyst Duct | qwen3.8-flash-next-iq3_xxs | **63.6 tok/s snitt (topp 68.8)** | 61.0°C | 65.0°C | ~128W snitt | +13.0°C (vid 78°C) | Bästa all-in-VRAM 64GB-profilen: ny 3D-printad duct + MTP `spec=3` sänkte temp från 74°C -> 65°C |
