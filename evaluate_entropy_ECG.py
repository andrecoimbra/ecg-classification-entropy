import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Função para cálculo da entropia
def Max_Entropy(x_rand, y_rand, Serie, StatsBlock):
    Threshold = 0.0
    Frac = 10
    Frac2 = 4
    Increase_Thr = 1.0 / Frac
    Max_Threshold = 0.0
    S_Max = 0
    StatsM = np.zeros((2 ** (StatsBlock * StatsBlock)))
    pow_vec = np.array([2**k for k in range(StatsBlock * StatsBlock)], dtype=np.int64)

    for i in range(0, Frac2):
        if i > 0:
            Threshold = Max_Threshold - Increase_Thr
            Increase_Thr = (2.0 * Increase_Thr) / Frac

        for j in range(0, Frac):
            Stats = np.zeros_like(StatsM)
            for count in range(len(x_rand)):
                Add = 0
                for count_y in range(StatsBlock):
                    for count_x in range(StatsBlock):
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
                    S -= (Hist_S / samples) * np.log(Hist_S / samples)

            if S > S_Max:
                S_Max = S
                Max_Threshold = Threshold
                StatsM = Stats.copy()

            Threshold += Increase_Thr

    return Max_Threshold, S_Max, StatsM


# Parâmetros globais
StatsBlock = 3
samples = 5

# Carregamento dos arquivos
with open("filename.dat", "r") as myfile:
    fn = myfile.read().splitlines()

with open("diagnostics.dat", "r") as myfile:
    di = myfile.read().splitlines()

with open("block_list.dat", "r") as myfile:
    block_list = myfile.read().splitlines()

# Lista de classes e mapeamento nome → índice
diag_list = np.array(
    ["SR", "SB", "AFIB", "ST", "SVT", "AF", "SI", "AT", "AVNRT", "AVRT", "SAAWR"]
)
diag_map = {label: idx for idx, label in enumerate(diag_list)}


# Função paralelizável
def process_file(i):
    filename = fn[i]
    if filename in block_list:
        return None

    label_str = di[i]
    label_idx = diag_map[label_str]

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

    return Aux, label_idx


# Execução paralela
def main():
    start_time = time.time()
    diag_counts = np.zeros(len(diag_list), dtype=int)

    X = [None] * len(fn)  # Reservar espaço
    Y = [None] * len(fn)

    print("Computing microstate entropy in parallel (ordered)...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, i): i for i in range(len(fn))}

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Entropy calculation"
        ):
            i = futures[future]
            result = future.result()
            if result is not None:
                Aux, label_idx = result
                X[i] = Aux
                Y[i] = label_idx
                diag_counts[label_idx] += 1

    # Filtrar resultados válidos (excluindo blocos ignorados)
    X = np.array([x for x in X if x is not None])
    Y = np.array([y for y in Y if y is not None])

    print("\nRhythm distribution:")
    for label, count in zip(diag_list, diag_counts):
        print(f"{label:7} → {count} amostras")

    np.save(f"{samples}_Data_S.npy", X)
    np.save(f"{samples}_Data_L.npy", Y)
    print(f"--- {(time.time() - start_time) / 60:.2f} minutes ---")


# Ponto de entrada
if __name__ == "__main__":
    main()
