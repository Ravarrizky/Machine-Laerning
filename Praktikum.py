import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, confusion_matrix)
from scipy.optimize import linear_sum_assignment

OUTPUT_DIR = "hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 65)
print(" CLUSTERING JENIS WINE DENGAN ALGORITMA K-MEANS")
print("=" * 65)

# 1. PENGAMBILAN DATASET
print("\n[1/5] Memuat dataset Wine...")
wine = load_wine(as_frame=True)
X = wine.data
y_true = wine.target          # hanya untuk validasi, TIDAK dipakai saat clustering
target_names = list(wine.target_names)
print(f"Dimensi data: {X.shape}")
print("Missing value:", int(X.isnull().sum().sum()))

# 2. EDA
print("\n[2/5] Exploratory Data Analysis...")
print(X.describe().T[['mean', 'std', 'min', 'max']].round(2))
counts = pd.Series(y_true).map(dict(enumerate(target_names))).value_counts().sort_index()
print("\nDistribusi label asli (hanya referensi):")
print(counts)

# Grafik 1: Korelasi fitur
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Heatmap Korelasi Antar Fitur Dataset Wine", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_korelasi_fitur.png"), dpi=200)
plt.close()

# 3. PREPROCESSING
print("\n[3/5] Preprocessing (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PENENTUAN K OPTIMAL
print("\n[4/5] Mencari jumlah cluster optimal (K=2..10)...")
ks = list(range(2, 11))
inertia, sil, dbi = [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))
    dbi.append(davies_bouldin_score(X_scaled, km.labels_))
    print(f"K={k:2d} | Inertia={km.inertia_:8.2f} | Silhouette={sil[-1]:.4f} | DBI={dbi[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(ks, inertia, "o-", color="#3b82f6")
axes[0].axvline(3, ls="--", color="#ef4444")
axes[0].set_title("Elbow Method (Inertia)", fontweight="bold")
axes[0].set_xlabel("Jumlah Cluster (K)"); axes[0].set_ylabel("Inertia")
axes[1].plot(ks, sil, "o-", color="#10b981")
axes[1].axvline(3, ls="--", color="#ef4444")
axes[1].set_title("Silhouette Score", fontweight="bold")
axes[1].set_xlabel("Jumlah Cluster (K)"); axes[1].set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_elbow_silhouette.png"), dpi=200)
plt.close()

# 5. MODEL FINAL & PERBANDINGAN DENGAN/TANPA SCALING
print("\n[5/5] Model final K=3 & perbandingan dengan/tanpa scaling...")

def evaluasi(data, labels):
    return {
        "Silhouette": silhouette_score(data, labels),
        "Davies-Bouldin": davies_bouldin_score(data, labels),
        "Calinski-Harabasz": calinski_harabasz_score(data, labels),
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
    }

km_a = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X.values)       # Model A: tanpa scaling
km_b = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)       # Model B: dengan scaling
res_a = evaluasi(X.values, km_a.labels_)
res_b = evaluasi(X_scaled, km_b.labels_)

print("\n" + "=" * 70)
print(f"{'METRIK':<22} | {'MODEL A (TANPA SCALING)':<24} | {'MODEL B (SCALING)':<18}")
print("=" * 70)
for m in res_a:
    print(f"{m:<22} | {res_a[m]:<24.4f} | {res_b[m]:<18.4f}")
print("=" * 70)

def map_cluster_to_label(y_true, labels):
    cm = confusion_matrix(y_true, labels)
    r, c = linear_sum_assignment(-cm)
    mapping = {c_i: r_i for r_i, c_i in zip(r, c)}
    return np.array([mapping[l] for l in labels])

pred_a = map_cluster_to_label(y_true, km_a.labels_)
pred_b = map_cluster_to_label(y_true, km_b.labels_)
cm_a = confusion_matrix(y_true, pred_a)
cm_b = confusion_matrix(y_true, pred_b)
purity_a = np.trace(cm_a) / cm_a.sum()
purity_b = np.trace(cm_b) / cm_b.sum()
print(f"\nCluster purity A: {purity_a*100:.2f}% | B: {purity_b*100:.2f}%")
print("Confusion matrix A:\n", cm_a)
print("Confusion matrix B:\n", cm_b)
print("Ukuran cluster B:", np.bincount(km_b.labels_))

# Profil cluster (nilai rata-rata asli)
prof = X.copy(); prof["cluster"] = pred_b
profil = prof.groupby("cluster").mean().round(2)
profil.index = target_names
print("\nProfil rata-rata fitur kunci per cluster (Model B):")
print(profil[["alcohol", "flavanoids", "color_intensity", "proline", "total_phenols"]])

# Grafik 3: Visualisasi PCA
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X_scaled)
print("\nExplained variance PCA 2D:", pca.explained_variance_ratio_.round(4),
      "total", round(pca.explained_variance_ratio_.sum(), 4))
cent = pca.transform(km_b.cluster_centers_)
palette = ["#10b981", "#ef4444", "#f59e0b"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, name in enumerate(target_names):
    axes[0].scatter(Z[y_true == i, 0], Z[y_true == i, 1], s=35, c=palette[i],
                    edgecolor="black", linewidth=.4, label=name)
axes[0].set_title("Label Asli (Referensi)", fontweight="bold")
for i in range(3):
    axes[1].scatter(Z[pred_b == i, 0], Z[pred_b == i, 1], s=35, c=palette[i],
                    edgecolor="black", linewidth=.4, label=f"Cluster {i} ({target_names[i]})")
# pusat cluster (urutan sesuai mapping)
cm_map = confusion_matrix(y_true, km_b.labels_)
r, c = linear_sum_assignment(-cm_map)
for r_i, c_i in zip(r, c):
    axes[1].scatter(cent[c_i, 0], cent[c_i, 1], marker="X", s=250, c="black", edgecolor="white", linewidth=1.5)
axes[1].set_title("Hasil K-Means (K=3) + Centroid (X)", fontweight="bold")
for ax in axes:
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_visualisasi_cluster_pca.png"), dpi=200)
plt.close()

# Grafik 4: Perbandingan metrik
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
mm = ["ARI", "NMI", "Silhouette"]
x = np.arange(len(mm)); w = .35
axes[0].bar(x - w/2, [res_a[m] for m in mm], w, label="A (Tanpa Scaling)", color="#3b82f6", edgecolor="black")
axes[0].bar(x + w/2, [res_b[m] for m in mm], w, label="B (StandardScaler)", color="#10b981", edgecolor="black")
axes[0].set_xticks(x); axes[0].set_xticklabels(mm); axes[0].set_ylim(0, 1)
axes[0].set_title("Perbandingan Metrik Clustering", fontweight="bold"); axes[0].legend()
sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=[f"C{i}" for i in range(3)], yticklabels=target_names)
axes[1].set_title("Cluster vs Label Asli: Model A", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_perbandingan_metrik.png"), dpi=200)
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens", cbar=False,
            xticklabels=[f"C{i}" for i in range(3)], yticklabels=target_names)
plt.title("Cluster vs Label Asli: Model B", fontweight="bold")
plt.ylabel("Label Asli"); plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_crosstab_model_b.png"), dpi=200)
plt.close()
print("\nSelesai! Grafik tersimpan di ./hasil/")
