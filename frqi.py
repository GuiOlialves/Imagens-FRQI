import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ── 1. CARREGA IMAGEM ─────────────────────────────────────────────────────────
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

INDICE = 15
imagem = x_train[INDICE] / 255.0
label  = y_train[INDICE]
imagem_8x8 = resize(imagem, (8, 8), anti_aliasing=True)
pixels = imagem_8x8.flatten()

print(f"Dígito: {label}")

# ── 2. PARÂMETROS ─────────────────────────────────────────────────────────────
n_pos      = 6
n_total    = n_pos + 1
qubit_cor  = 0
qubits_pos = list(range(1, n_total))
SHOTS      = 6400

# ── 3. CIRCUITO FRQI ──────────────────────────────────────────────────────────
def circuito_frqi(pixels):
    qc = QuantumCircuit(n_total)
    for q in qubits_pos:
        qc.h(q)
    for i, pixel in enumerate(pixels):
        theta = float(pixel) * (np.pi / 2)
        if theta < 1e-6:
            continue
        bin_i = format(i, f'0{n_pos}b')
        for j, bit in enumerate(bin_i):
            if bit == '0':
                qc.x(qubits_pos[j])
        qc.mcry(theta, qubits_pos, qubit_cor)
        for j, bit in enumerate(bin_i):
            if bit == '0':
                qc.x(qubits_pos[j])
    return qc

# ── 4. CRIPTOGRAFIA ───────────────────────────────────────────────────────────
CHAVE = np.array([0.7, 1.2, 0.4, 1.8, 0.9, 0.3])

def circuito_criptografia(chave):
    qc = QuantumCircuit(n_total)
    for j, q in enumerate(qubits_pos):
        qc.rz(chave[j], q)
    for j in range(len(qubits_pos) - 1):
        qc.cx(qubits_pos[j], qubits_pos[j+1])
    for j, q in enumerate(qubits_pos):
        qc.rz(chave[j] * 1.5, q)
    return qc

def circuito_descriptografia(chave):
    return circuito_criptografia(chave).inverse()

# ── 5. MONTA CIRCUITOS ────────────────────────────────────────────────────────
qc_original = circuito_frqi(pixels)
qc_original.measure_all()

qc_criptografado = circuito_frqi(pixels)
qc_criptografado.compose(circuito_criptografia(CHAVE), inplace=True)
qc_criptografado.measure_all()

qc_recuperado = circuito_frqi(pixels)
qc_recuperado.compose(circuito_criptografia(CHAVE), inplace=True)
qc_recuperado.compose(circuito_descriptografia(CHAVE), inplace=True)
qc_recuperado.measure_all()

# ── 6. EXECUTA ────────────────────────────────────────────────────────────────
sim = AerSimulator()
print("Executando circuitos...")

t0 = time.time()
counts_original      = sim.run(qc_original,      shots=SHOTS).result().get_counts()
t_original = time.time() - t0

t0 = time.time()
counts_criptografado = sim.run(qc_criptografado, shots=SHOTS).result().get_counts()
t_cripto = time.time() - t0

t0 = time.time()
counts_recuperado    = sim.run(qc_recuperado,    shots=SHOTS).result().get_counts()
t_recuperado = time.time() - t0

print(f"Original:      {t_original:.4f}s")
print(f"Criptografado: {t_cripto:.4f}s")
print(f"Recuperado:    {t_recuperado:.4f}s")

# ── 7. RECONSTRUÇÃO ───────────────────────────────────────────────────────────
def reconstruir_imagem(counts, n_pos=6):
    """Retorna imagem 8x8 com pixels normalizados 0-1."""
    n_pixels       = 2 ** n_pos
    contagem_1     = np.zeros(n_pixels)
    contagem_total = np.zeros(n_pixels)
    for estado, contagem in counts.items():
        bits = estado.replace(' ', '')
        cor  = int(bits[-1])
        pos  = int(bits[:-1], 2)
        if pos < n_pixels:
            contagem_total[pos] += contagem
            if cor == 1:
                contagem_1[pos] += contagem
    imagem_rec = np.zeros(n_pixels)
    for i in range(n_pixels):
        if contagem_total[i] > 0:
            prob          = np.clip(contagem_1[i] / contagem_total[i], 0, 1)
            theta         = np.arcsin(np.sqrt(prob))
            imagem_rec[i] = theta / (np.pi / 2)
    return imagem_rec.reshape(8, 8)

def reconstruir_angulos(counts, n_pos=6):
    """Retorna matriz 8x8 com ângulos θ em radianos (0 a π/2) por pixel."""
    n_pixels       = 2 ** n_pos
    contagem_1     = np.zeros(n_pixels)
    contagem_total = np.zeros(n_pixels)
    for estado, contagem in counts.items():
        bits = estado.replace(' ', '')
        cor  = int(bits[-1])
        pos  = int(bits[:-1], 2)
        if pos < n_pixels:
            contagem_total[pos] += contagem
            if cor == 1:
                contagem_1[pos] += contagem
    angulos = np.zeros(n_pixels)
    for i in range(n_pixels):
        if contagem_total[i] > 0:
            prob       = np.clip(contagem_1[i] / contagem_total[i], 0, 1)
            angulos[i] = np.arcsin(np.sqrt(prob))  # θ em radianos
    return angulos.reshape(8, 8)

img_original      = reconstruir_imagem(counts_original)
img_criptografada = reconstruir_imagem(counts_criptografado)
img_recuperada    = reconstruir_imagem(counts_recuperado)

ang_original      = reconstruir_angulos(counts_original)
ang_criptografada = reconstruir_angulos(counts_criptografado)
ang_recuperada    = reconstruir_angulos(counts_recuperado)

# ângulos teóricos: o que foi codificado (sem ruído estatístico)
ang_teorico = (imagem_8x8 * (np.pi / 2)).reshape(8, 8)

# ── 8. MÉTRICAS ───────────────────────────────────────────────────────────────
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b, max_val=1.0):
    m = mse(a, b)
    return float('inf') if m == 0 else 10 * np.log10(max_val**2 / m)

print(f"\nMSE  original  vs 8x8 real:  {mse(imagem_8x8, img_original):.6f}")
print(f"MSE  recuperado vs 8x8 real: {mse(imagem_8x8, img_recuperada):.6f}")
print(f"MSE  criptograf vs 8x8 real: {mse(imagem_8x8, img_criptografada):.6f}")
print(f"PSNR original:   {psnr(imagem_8x8, img_original):.2f} dB")
print(f"PSNR recuperado: {psnr(imagem_8x8, img_recuperada):.2f} dB")

# ── 9. MATRIZES DE ÂNGULOS ────────────────────────────────────────────────────
np.set_printoptions(precision=4, suppress=True, linewidth=100)

print("\n── Ângulos reconstruídos (FRQI) — radianos ──")
print(ang_original)

print("\n── Ângulos recuperados (após descriptografia) — radianos ──")
print(ang_recuperada)

# ── 10. VISUALIZAÇÃO ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].imshow(imagem_8x8,        cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f"Original 8x8\n(dígito: {label})")
axes[0].axis('off')

axes[1].imshow(img_original,      cmap='gray', vmin=0, vmax=1)
#axes[1].set_title(f"FRQI reconstruído\nPSNR: {psnr(imagem_8x8, img_original):.1f}dB")
axes[1].axis('off')

axes[2].imshow(img_criptografada, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Criptografado\n(chave aplicada)")
axes[2].axis('off')

axes[3].imshow(img_recuperada,    cmap='gray', vmin=0, vmax=1)
#axes[3].set_title(f"Recuperado\nPSNR: {psnr(imagem_8x8, img_recuperada):.1f}dB")
axes[3].axis('off')

plt.suptitle("FRQI — Criptografia Quântica de Imagem", fontsize=13)
plt.tight_layout()
plt.savefig('frqi_angulos.png', dpi=150, bbox_inches='tight')
plt.show()