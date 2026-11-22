# Custom 3D-Printed 120mm Radial Fan Duct for Dual NVIDIA Tesla V100 (HP Z8 G4)

Passively cooled datacenter **NVIDIA Tesla V100 PCIe 32GB** GPUs have dense, high-resistance fin stacks designed for screaming 15,000 RPM 40mm server fans. Small axial fans cannot build enough static pressure without sounding like a jet engine.

This custom 3D-printed dual-card fan duct solves the problem by channeling a **single 120mm radial blower fan** (plus the HP Z8 G4 chassis airflow) directly through both adjacent Tesla V100 heatsinks at **only 25% fan speed** (`Increase Idle Fan Speed(%) = 25` in HP BIOS).

---

## Thermal & Acoustic Results at Only 25% Fan Speed

* **Whisper-Quiet Operation:** Running a 120mm radial fan at **25% duty cycle** provides high static pressure with minimal acoustic noise—quiet enough to sit next to your desk.
* **Idle GPU Temperatures (`135 MHz` Eco Mode):** **`38°C`** on both GPU 0 and GPU 1 (`~24–26W` draw).
* **Sustained Inference Load (`140W` Cap, `1380 MHz` Boost, 111GB `UD-Q4_K_XL`):**
  * **Single-Stream (`77.7–82.0 tok/s`):** **`53°C`** on GPU 0 / **`52°C`** on GPU 1.
  * **Dual-Stream Parallel Batching (`87.5–93.6 tok/s`):** **`54°C`** on GPU 0 / **`55°C`** on GPU 1.
* **Onboard NVMe SSD Protection:** Moving from 20% to 25% BIOS fan floor drops the onboard Samsung NVMe PCIe controller (`Sensor 2`) by **`-47.9°C`** (from `98.8°C` down to `50.9°C`) and NAND flash (`Composite`) to **`37.9°C`**.

---

## Included 3D Printable Files (`.stl` & `.3mf`)

All 3D print files are located right in this folder ([`hardware/fan-duct/`](./)):

| File | Description |
| :--- | :--- |
| **[`v100-dual-duct-120mm-radial-fan-v2.stl`](./v100-dual-duct-120mm-radial-fan-v2.stl)** | **Main 120mm Radial Fan Dual-V100 Shroud Duct (v2):** Manifolds a single 120mm radial fan into both adjacent Tesla V100 PCIe intake ports with zero pressure leak. |
| **[`v100-fan-channel-hp-z8-g4-dual-card.3mf`](./v100-fan-channel-hp-z8-g4-dual-card.3mf)** | **HP Z8 G4 Dual-Card Channel Project (`.3mf`):** Ready-to-slice 3MF project with orientation and wall parameters for the HP Z8 G4 PCIe bay. |
| **[`v100-fan-channel-hp-z8-g4-dual-card.stl`](./v100-fan-channel-hp-z8-g4-dual-card.stl)** | **HP Z8 G4 Dual-Card Channel Mesh (`.stl`):** Raw STL mesh of the HP Z8 G4 dual-card air channel. |

### Printing Recommendations
* **Material:** **PETG**, **ABS**, or **ASA** (recommended over PLA so the shroud never softens near the PCIe exhaust zone).
* **Walls / Perimeters:** 3–4 perimeters, 20–25% gyroid or grid infill.
