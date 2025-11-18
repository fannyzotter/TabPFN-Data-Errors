import re
import numpy as np
import matplotlib.pyplot as plt

ground_temp = 20.0
Faktor = 0.0416754702
Offset = -7374.979078
offset_temp = 32.0

def block_mean(seq, block_size=100, include_partial=False):
    """Mittelwerte in nicht überlappenden Blöcken.
    include_partial=False: Rest am Ende wird verworfen.
    """
    n = len(seq)
    usable = (n // block_size) * block_size if not include_partial else n
    if usable == 0:
        return np.array([])
    seq = np.array(seq[:usable], dtype=float)
    # In Blöcke formen und entlang der Zeilen mitteln
    num_blocks = math.ceil(usable / block_size) if include_partial else usable // block_size
    pad = num_blocks * block_size - usable
    if include_partial and pad:
        seq = np.pad(seq, (0, pad), constant_values=np.nan)
    return np.nanmean(seq.reshape(num_blocks, block_size), axis=1)


def calibrated_weight_for_temperature(hx_value, temp):
    weight = Faktor * hx_value + Offset
    # Temperaturkorrektur
    temp_diff = temp - ground_temp
    weight_corrected = weight - (temp_diff * offset_temp)  # Ann
    return weight, weight_corrected

# Pfad zur Textdatei
file_path1 = "code/esp_log4.txt"
file_path2 = "code/esp_log5.txt"
file_path3 = "code/esp_log7.txt"

file_path4 = "code/esp_log_der12_11um7_02.txt"
file_path5 = "code/esp_log_der16_11um10_11.txt"



# Listen für Messdaten
hx_values = []
weights = []
temperatures = []

# Datei einlesen und Werte extrahieren
with open(file_path1, "r", encoding="utf-8") as f:
    for line in f:
        hx_match = re.search(r"HX711:\s*(\d+)", line)
        weight_match = re.search(r"weight:\s*([0-9.]+)", line)
        temp_match = re.search(r"Temperatur:\s*([0-9.]+)", line)
        
        if hx_match:
            hx_values.append(int(hx_match.group(1)))
        if temp_match:
            temperatures.append(float(temp_match.group(1)))

with open(file_path2, "r", encoding="utf-8") as f:
    for line in f:
        hx_match = re.search(r"HX711:\s*(\d+)", line)
        weight_match = re.search(r"weight:\s*([0-9.]+)", line)
        temp_match = re.search(r"Temperatur:\s*([0-9.]+)", line)
        
        if hx_match:
            hx_values.append(int(hx_match.group(1)))
        if temp_match:
            temperatures.append(float(temp_match.group(1)))

with open(file_path3, "r", encoding="utf-8") as f:
    for line in f:
        hx_match = re.search(r"HX711:\s*(\d+)", line)
        weight_match = re.search(r"weight:\s*([0-9.]+)", line)
        temp_match = re.search(r"Temperatur:\s*([0-9.]+)", line)
        
        if hx_match:
            hx_values.append(int(hx_match.group(1)))
        if temp_match:
            temperatures.append(float(temp_match.group(1)))

with open(file_path4, "r", encoding="utf-8") as f:
    for line in f:
        hx_match = re.search(r"HX711:\s*(\d+)", line)
        weight_match = re.search(r"weight:\s*([0-9.]+)", line)
        temp_match = re.search(r"Temperatur:\s*([0-9.]+)", line)
        
        if hx_match:
            hx_values.append(int(hx_match.group(1)))
        if temp_match:
            temperatures.append(float(temp_match.group(1)))

with open(file_path5, "r", encoding="utf-8") as f:
    for line in f:
        hx_match = re.search(r"HX711:\s*(\d+)", line)
        weight_match = re.search(r"weight:\s*([0-9.]+)", line)
        temp_match = re.search(r"Temperatur:\s*([0-9.]+)", line)
        
        if hx_match:
            hx_values.append(int(hx_match.group(1)))
        if temp_match:
            temperatures.append(float(temp_match.group(1)))

min_len = min(len(hx_values), len(temperatures))
hx_values = hx_values[:min_len]
temperatures = temperatures[:min_len]

# --- HIER ANPASSEN: Blockgröße ---
BLOCK = 1000

# HX711 blockgemittelt
hx_values_block = block_mean(hx_values, BLOCK, include_partial=False)

temperatures_block = block_mean(temperatures, BLOCK, include_partial=False)

weights_block, temp_corrected_weight = calibrated_weight_for_temperature(np.array(hx_values_block), np.array(temperatures_block))

x = range(len(hx_values_block))

# Plot erstellen
fig, ax1 = plt.subplots()

# HX711-Werte (blau)
ax1.set_xlabel("Messblock")
ax1.set_ylabel("Gewicht (g)", color="tab:blue")
ax1.plot(range(len(temp_corrected_weight)), temp_corrected_weight, color="tab:blue", alpha=0.4, label="Gewicht (g), korrigiert")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Gewicht (grün, geglättet)
#ax1.plot(x, hx_values_block, label=f"HX711 (Blockmittel {BLOCK})", alpha=0.8)
ax1.plot(x, weights_block, linestyle="--", label=f"Gewicht (g), Blockgröße {BLOCK*2} Sekunden Messung", alpha=0.8)

# Temperatur (rot, rechte Achse)
ax2 = ax1.twinx()
ax2.set_ylabel("Temperatur (°C)", color="tab:red")
ax2.plot(x, temperatures_block, color="tab:red", label=f"Temperatur (°C), Blockgröße {BLOCK*2} Sekunden Messung", alpha=0.8)
ax2.tick_params(axis="y", labelcolor="tab:red")

# Titel und Legende
fig.suptitle("HX711-, Gewicht- und Temperaturverlauf (geglättet)")
fig.legend(loc="upper left")
plt.tight_layout()
plt.show()