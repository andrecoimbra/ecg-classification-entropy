import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

start_time = time.time()

with open("filename.dat", "r") as myfile:
    fn = myfile.read().splitlines()

with open("diagnostics.dat", "r") as myfile:
    di = myfile.read().splitlines()

with open("block_list.dat", "r") as myfile:
    block_list = myfile.read().splitlines()

samples = 5


def Max_Entropy(x_rand, y_rand, Serie, StatsBlock):
    Threshold = 0.0
    Frac = 10
    Frac2 = 4
    Increase_Thr = 1.0 / Frac
    Max_Threshold = 0.0
    S_Max = 0
    S = 0
    StatsM = np.zeros((pow(2, StatsBlock * StatsBlock)))
    pow_vec = np.zeros((StatsBlock * StatsBlock), np.int64)

    for K in range(0, (StatsBlock * StatsBlock)):
        pow_vec[K] = int(pow(2, K))

    for i in range(0, Frac2):

        if i > 0:
            Threshold = Max_Threshold - 1.0 * Increase_Thr
            Increase_Thr = (2.0 * Increase_Thr) / (1.0 * Frac)

        for j in range(0, Frac):

            Stats = np.zeros((pow(2, StatsBlock * StatsBlock)))
            for count in range(len(x_rand)):
                Add = 0
                for count_y in range(0, StatsBlock):
                    for count_x in range(0, StatsBlock):
                        a = int(
                            abs(
                                Serie[(x_rand[count] + count_x)]
                                - Serie[(y_rand[count] + count_y)]
                            )
                            <= Threshold
                        )
                        Add += a * pow_vec[count_x + count_y * StatsBlock]
                Stats[Add] += 1
            S = 0
            for Hist_S in Stats:
                if Hist_S > 0:
                    S -= (float(Hist_S) / (1.0 * samples)) * (
                        np.log((float(Hist_S) / (1.0 * samples)))
                    )

            if S > S_Max:
                S_Max = S
                Max_Threshold = Threshold
                StatsM = Stats

            Threshold = Threshold + Increase_Thr
    return Max_Threshold, S_Max, StatsM


StatsBlock = 3

# Pré-cálculo do vetor de potências (fixo por StatsBlock)
pow_vec = np.array([2**k for k in range(StatsBlock * StatsBlock)], dtype=np.int64)

# Lista de classes e mapeamento nome → índice
diag_list = np.array(
    ["SR", "SB", "AFIB", "ST", "SVT", "AF", "SI", "AT", "AVNRT", "AVRT", "SAAWR"]
)
# Mapeia os rótulos para índices uma única vez
diag_map = {label: idx for idx, label in enumerate(diag_list)}

diag_counts = np.zeros(len(diag_list), dtype=int)

# Vetores finais
X = []
Y = []

print("Computing microstate entropy...")
for i, filename in enumerate(tqdm(fn, desc="Entropy calculation")):
    if filename in block_list:
        continue

    label_str = di[i]
    label_idx = diag_map[label_str]

    # print(i, label_str, label_idx)
    diag_counts[label_idx] += 1

    # Carregar os dados diretamente com numpy (mais rápido)
    filepath = Path(f"database/ECGDataDenoised/{filename}.csv")
    data = np.loadtxt(filepath, delimiter=",")  # shape: (time, leads)

    Aux = []
    for k in range(data.shape[1]):
        Serie = data[:, k].astype(np.float64)
        Serie = (Serie - Serie.min()) / (Serie.max() - Serie.min())

        Size = len(Serie)
        x_rand = np.random.choice(Size - StatsBlock - 1, samples)
        y_rand = np.random.choice(Size - StatsBlock - 1, samples)

        Eps, S_max, Stats = Max_Entropy(x_rand, y_rand, Serie, StatsBlock)
        Aux.append(S_max / (StatsBlock * StatsBlock * np.log(2)))
        Aux.append(Eps)

    X.append(Aux)
    Y.append(label_idx)

X = np.array(X)
Y = np.array(Y)

print("\nRhythm distribution:")
for label, count in zip(diag_list, diag_counts):
    print(f"{label:7} → {count} amostras")


np.save(f"{samples}_Data_S.npy", X)
np.save(f"{samples}_Data_L.npy", Y)
print(f"--- {(time.time() - start_time) / 60:.2f} minutes ---")
