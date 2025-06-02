import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import antropy as ant
from scipy.stats import entropy as shannon_entropy


# Function for calculating microstate entropy
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


# Global parameters
StatsBlock = 3
samples = 10000

# File loading
with open("filename.dat", "r") as myfile:
    fn = myfile.read().splitlines()

with open("diagnostics.dat", "r") as myfile:
    di = myfile.read().splitlines()

with open("block_list.dat", "r") as myfile:
    block_list = myfile.read().splitlines()

# List of classes and name → index mapping
diag_list = np.array(
    ["SR", "SB", "AFIB", "ST", "SVT", "AF", "SI", "AT", "AVNRT", "AVRT", "SAAWR"]
)
diag_map = {label: idx for idx, label in enumerate(diag_list)}


def process_file(i):
    filename = fn[i]
    if filename in block_list:
        return None

    label_str = di[i]
    label_idx = diag_map[label_str]

    filepath = Path(f"database/ECGDataDenoised/{filename}.csv")
    data = np.loadtxt(filepath, delimiter=",")  # shape: (time, leads)

    # Initialize separate lists for each entropy measure
    rec_en_list = []
    eps_list = []
    apen_list = []
    sampen_list = []
    shannon_list = []
    spec_list = []
    svd_list = []

    for k in range(data.shape[1]):
        Serie = data[:, k].astype(np.float64)
        if Serie.max() == Serie.min():
            continue  # skip normalization to avoid division by zero

        Serie = (Serie - Serie.min()) / (Serie.max() - Serie.min())

        Size = len(Serie)
        x_rand = np.random.choice(Size - StatsBlock - 1, samples)
        y_rand = np.random.choice(Size - StatsBlock - 1, samples)

        # Recurrence entropy
        Eps, S_max, Stats = Max_Entropy(x_rand, y_rand, Serie, StatsBlock)
        rec_en = S_max / (StatsBlock * StatsBlock * np.log(2))

        # Other entropy measures
        apen = ant.app_entropy(Serie)
        sampen = ant.sample_entropy(Serie)
        hist, _ = np.histogram(Serie, bins=20, density=True)
        shannon = shannon_entropy(hist + 1e-12)
        spec = ant.spectral_entropy(Serie, sf=500, normalize=True)
        svd = ant.svd_entropy(Serie, normalize=True)

        # Append each measure to its corresponding list
        rec_en_list.append(rec_en)
        eps_list.append(Eps)
        apen_list.append(apen)
        sampen_list.append(sampen)
        shannon_list.append(shannon)
        spec_list.append(spec)
        svd_list.append(svd)

    # Concatenate all lists in the desired order
    Aux = (
        rec_en_list
        + eps_list
        + apen_list
        + sampen_list
        + shannon_list
        + spec_list
        + svd_list
    )

    return Aux, label_idx


# Parallel execution
def main():
    diag_counts = np.zeros(len(diag_list), dtype=int)
    X = [None] * len(fn)
    Y = [None] * len(fn)

    print("Computing all entropy measures in parallel (ordered)...")
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

    # Filter valid results
    X = np.array([x for x in X if x is not None])
    Y = np.array([y for y in Y if y is not None])

    print("\nRhythm distribution:")
    for label, count in zip(diag_list, diag_counts):
        print(f"{label:7} → {count} samples")

    np.save(f"Data_S_full.npy", X)
    np.save(f"Data_L_full.npy", Y)


# Entry point
if __name__ == "__main__":
    main()
