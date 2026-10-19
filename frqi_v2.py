"""FRQI v2: imagem classica -> estado quantico -> imagem reconstruida.

O arquivo frqi.py permanece como versao original do experimento.
Exemplo sem download: python frqi_v2.py --demo --sem-janela
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from skimage.transform import resize


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def validar_imagem(imagem: np.ndarray) -> np.ndarray:
    """Aceita uma matriz normalizada com dimensoes que sao potencias de dois."""
    imagem = np.asarray(imagem, dtype=float)
    if imagem.ndim != 2 or any(d == 0 or d & (d - 1) for d in imagem.shape):
        raise ValueError("A imagem deve ser 2D, com dimensoes potencias de dois.")
    if not np.all(np.isfinite(imagem)) or np.any((imagem < 0) | (imagem > 1)):
        raise ValueError("As intensidades devem ser finitas e estar entre 0 e 1.")
    return imagem


def circuito_frqi(imagem: np.ndarray) -> QuantumCircuit:
    """Codifica cos(theta)|0> + sin(theta)|1> para cada posicao do pixel.

    theta = intensidade * pi/2. RY usa o meio angulo em suas amplitudes,
    portanto a rotacao fisica deve ser RY(2*theta).
    q[0] e a cor; q[j+1] recebe o bit j (menos significativo primeiro).
    """
    imagem = validar_imagem(imagem)
    n_pos = (imagem.size - 1).bit_length()
    qubits_pos = list(range(1, n_pos + 1))
    qc = QuantumCircuit(n_pos + 1)
    for qubit in qubits_pos:
        qc.h(qubit)
    for indice, pixel in enumerate(imagem.ravel()):
        if pixel == 0:
            continue
        theta = float(pixel) * np.pi / 2
        if not qubits_pos:
            qc.ry(2 * theta, 0)
            continue
        # O bit j de indice corresponde a q[j+1]. Isto combina com get_counts.
        mascara = [q for j, q in enumerate(qubits_pos) if not (indice >> j) & 1]
        for qubit in mascara:
            qc.x(qubit)
        qc.mcry(2 * theta, qubits_pos, 0, mode="noancilla")
        for qubit in mascara:
            qc.x(qubit)
    return qc


def _reconstruir_pesos(pesos: np.ndarray, formato: tuple[int, int]) -> np.ndarray:
    """Cada linha contem pesos de cor 0 e 1 para uma posicao."""
    totais = pesos.sum(axis=1)
    probabilidades = np.divide(
        pesos[:, 1], totais, out=np.full(len(totais), np.nan), where=totais > 0
    )
    angulos = np.arcsin(np.sqrt(np.clip(probabilidades, 0, 1)))
    return (2 * angulos / np.pi).reshape(formato)


def reconstruir_ideal(qc: QuantumCircuit, formato: tuple[int, int]) -> np.ndarray:
    """Referencia de simulacao: calcula probabilidades do estado, sem shots.

    Este acesso ao vetor de estado nao e uma leitura disponivel em hardware.
    O vetor e obtido do circuito, sem reutilizar os pixels de entrada.
    """
    n_pixels = math.prod(formato)
    probabilidades = Statevector.from_instruction(qc).probabilities()
    if len(probabilidades) != 2 * n_pixels:
        raise ValueError("O formato informado nao corresponde aos qubits do circuito.")
    # O indice da base e (posicao << 1) | cor, porque cor e q[0].
    return _reconstruir_pesos(probabilidades.reshape(n_pixels, 2), formato)


def reconstruir_imagem(
    counts: dict[str, int], formato: tuple[int, int] = (8, 8)
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna a imagem e o numero de amostras por pixel.

    Posicoes nao observadas recebem NaN, nunca sao confundidas com pixels pretos.
    Qiskit escreve q[n]...q[1]q[0]: o ultimo bit e a cor.
    """
    n_pixels = math.prod(formato)
    if n_pixels < 1 or n_pixels & (n_pixels - 1):
        raise ValueError("O numero de pixels deve ser uma potencia de dois.")
    n_qubits = (n_pixels - 1).bit_length() + 1
    pesos = np.zeros((n_pixels, 2), dtype=np.int64)
    for estado, contagem in counts.items():
        bits = estado.replace(" ", "")
        if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
            raise ValueError("Contagem com bits incompatíveis com o formato da imagem.")
        if not isinstance(contagem, (int, np.integer)) or contagem < 0:
            raise ValueError("As contagens devem ser numeros inteiros nao negativos.")
        valor = int(bits, 2)
        posicao, cor = valor >> 1, valor & 1
        pesos[posicao, cor] += contagem
    return _reconstruir_pesos(pesos, formato), pesos.sum(axis=1).reshape(formato)


def metricas(referencia: np.ndarray, reconstruida: np.ndarray) -> dict:
    """Nao atribui fidelidade global a uma imagem com pixels ainda desconhecidos."""
    referencia = np.asarray(referencia, dtype=float)
    reconstruida = np.asarray(reconstruida, dtype=float)
    if referencia.shape != reconstruida.shape:
        raise ValueError("As imagens comparadas devem ter o mesmo formato.")
    cobertura = float(np.mean(np.isfinite(reconstruida)))
    resultado = {"cobertura": cobertura, "mse": None, "psnr_db": None,
                 "erro_maximo": None, "psnr_infinito": False}
    if cobertura < 1:
        return resultado
    diferenca = referencia - reconstruida
    mse = float(np.mean(diferenca ** 2))
    resultado.update(
        mse=mse,
        psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
        erro_maximo=float(np.max(np.abs(diferenca))),
        psnr_infinito=mse == 0,
    )
    return resultado


def executar_experimento(imagem: np.ndarray, shots: int = 6400, seed: int = 42) -> dict:
    """Simula a codificacao FRQI com medicoes reprodutiveis e referencia ideal."""
    imagem = validar_imagem(imagem)
    if shots <= 0:
        raise ValueError("O numero de shots deve ser positivo.")
    if seed < 0:
        raise ValueError("A semente deve ser nao negativa.")
    inicio = time.perf_counter()
    qc = circuito_frqi(imagem)
    ideal = reconstruir_ideal(qc, imagem.shape)
    tempo_ideal = time.perf_counter() - inicio

    qc_medido = qc.copy()
    qc_medido.measure_all()
    sim = AerSimulator(method="statevector")
    inicio = time.perf_counter()
    qc_compilado = transpile(qc_medido, sim, seed_transpiler=seed, optimization_level=1)
    tempo_transpilacao = time.perf_counter() - inicio
    inicio = time.perf_counter()
    counts = sim.run(qc_compilado, shots=shots, seed_simulator=seed).result().get_counts()
    tempo_simulacao = time.perf_counter() - inicio
    medida, amostras = reconstruir_imagem(counts, imagem.shape)
    return {
        "original": imagem.copy(), "ideal": ideal, "medida": medida,
        "amostras": amostras, "counts": dict(counts),
        "n_qubits": qc.num_qubits, "profundidade": qc_compilado.depth(),
        "shots": shots, "seed": seed,
        "tempo_ideal_s": tempo_ideal, "tempo_transpilacao_s": tempo_transpilacao,
        "tempo_simulacao_s": tempo_simulacao,
        "metricas_ideal": metricas(imagem, ideal),
        "metricas_medida": metricas(imagem, medida),
    }


def carregar_mnist(indice: int, arquivo: Path | None = None) -> tuple[np.ndarray, str]:
    """Usa o mesmo MNIST da v1, lendo o NPZ oficial sem precisar de TensorFlow."""
    if indice < 0:
        raise ValueError("O indice MNIST deve ser nao negativo.")
    caminho = arquivo if arquivo is not None else Path("dados/mnist.npz")
    if not caminho.exists():
        if arquivo is not None:
            raise FileNotFoundError(f"Arquivo MNIST nao encontrado: {caminho}")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        temporario = None
        try:
            print("Baixando MNIST do repositorio oficial do TensorFlow...")
            with urllib.request.urlopen(MNIST_URL, timeout=30) as resposta:
                with tempfile.NamedTemporaryFile(dir=caminho.parent, delete=False) as destino:
                    temporario = Path(destino.name)
                    shutil.copyfileobj(resposta, destino)
            temporario.replace(caminho)
        finally:
            if temporario is not None:
                temporario.unlink(missing_ok=True)
    with np.load(caminho, allow_pickle=False) as dados:
        if indice >= len(dados["x_train"]):
            raise ValueError("O indice informado ultrapassa o conjunto MNIST.")
        imagem = dados["x_train"][indice].astype(float) / 255
        label = int(dados["y_train"][indice])
    return imagem, f"MNIST indice {indice}, digito {label}"


def imagem_demo(lado: int = 8) -> np.ndarray:
    """Grade assimetrica com intensidades distintas, preto e branco nas extremidades."""
    return np.linspace(0, 1, lado * lado).reshape(lado, lado)


def salvar_resultados(resultado: dict, diretorio: Path, fonte: str, mostrar: bool) -> None:
    """Salva os dados numericos, as metricas e um painel de comparacao."""
    import matplotlib
    if not mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diretorio.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diretorio / "frqi_v2_imagens.npz", original=resultado["original"],
        ideal=resultado["ideal"], reconstruida=resultado["medida"],
        amostras_por_pixel=resultado["amostras"],
    )
    relatorio = {
        "fonte": fonte, "formato": list(resultado["original"].shape),
        "referencia": "Imagem em escala de cinza apos redimensionamento",
        "shots": resultado["shots"], "seed": resultado["seed"],
        "n_qubits": resultado["n_qubits"], "profundidade": resultado["profundidade"],
        "ideal": resultado["metricas_ideal"], "medicoes": resultado["metricas_medida"],
        "pixels_nao_medidos": int(np.sum(resultado["amostras"] == 0)),
        "amostras_minimas": int(resultado["amostras"].min()),
        "amostras_maximas": int(resultado["amostras"].max()),
        "tempos_s": {k.removeprefix("tempo_").removesuffix("_s"): resultado[k]
                     for k in ("tempo_ideal_s", "tempo_transpilacao_s", "tempo_simulacao_s")},
        "versoes": {p: version(p) for p in ("qiskit", "qiskit-aer", "numpy", "scikit-image")},
        "contagens": resultado["counts"],
    }
    (diretorio / "frqi_v2_metricas.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (diretorio / "frqi_v2_pixels.csv").open("w", encoding="utf-8", newline="") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["linha", "coluna", "original", "ideal", "reconstruida", "amostras"])
        for linha, coluna in np.ndindex(resultado["original"].shape):
            valor = resultado["medida"][linha, coluna]
            writer.writerow([linha, coluna, resultado["original"][linha, coluna],
                             resultado["ideal"][linha, coluna],
                             valor if np.isfinite(valor) else "", resultado["amostras"][linha, coluna]])

    lado = resultado["original"].shape
    erro = np.abs(resultado["original"] - resultado["medida"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    titulos = [f"Original {lado[0]} x {lado[1]}", "FRQI ideal\n(referencia do simulador)",
               f"Reconstruida\n({resultado['shots']} medicoes)", "Erro absoluto"]
    # Magenta indica uma posicao desconhecida por falta de medicoes.
    cinza = plt.colormaps["gray"].with_extremes(bad="#dc22a7")
    for ax, dados, titulo in zip(axes[:3],
                                (resultado["original"], resultado["ideal"], resultado["medida"]),
                                titulos[:3]):
        ax.imshow(dados, cmap=cinza, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(titulo)
        ax.axis("off")
    mapa = axes[3].imshow(erro, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(titulos[3])
    axes[3].axis("off")
    fig.colorbar(mapa, ax=axes[3], shrink=0.7)
    fig.suptitle("FRQI v2 - codificacao e reconstrucao de imagem")
    fig.savefig(diretorio / "frqi_v2_comparacao.png", dpi=150)
    if mostrar:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--imagem", type=Path, help="Arquivo de imagem a reconstruir.")
    grupo.add_argument("--mnist-indice", type=int, help="Indice MNIST (padrao: 15).")
    grupo.add_argument("--demo", action="store_true", help="Grade assimetrica sem download.")
    parser.add_argument("--mnist-arquivo", type=Path, help="MNIST NPZ ja baixado, para uso offline.")
    parser.add_argument("--lado", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--shots", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path, default=Path("resultados_v2"))
    parser.add_argument("--sem-janela", action="store_true", help="Apenas salvar os resultados.")
    args = parser.parse_args()
    try:
        if args.shots <= 0 or args.seed < 0:
            raise ValueError("Use shots positivos e uma semente nao negativa.")
        if args.demo:
            imagem, fonte = imagem_demo(args.lado), "Grade assimetrica de demonstracao"
        elif args.imagem:
            with Image.open(args.imagem) as entrada:
                imagem = np.asarray(ImageOps.exif_transpose(entrada).convert("L"), dtype=float) / 255
            fonte = str(args.imagem)
        else:
            imagem, fonte = carregar_mnist(
                args.mnist_indice if args.mnist_indice is not None else 15, args.mnist_arquivo
            )
        imagem = resize(imagem, (args.lado, args.lado), anti_aliasing=True)
        print(f"Fonte: {fonte}")
        print(f"Referencia para comparacao: {args.lado} x {args.lado} pixels em tons de cinza")
        resultado = executar_experimento(imagem, args.shots, args.seed)
        for nome, chave in (("Ideal", "metricas_ideal"), ("Medicoes", "metricas_medida")):
            m = resultado[chave]
            if m["mse"] is None:
                print(f"{nome}: cobertura {100 * m['cobertura']:.1f}%; fidelidade global indefinida")
            else:
                psnr = "infinito" if m["psnr_infinito"] else f"{m['psnr_db']:.2f} dB"
                print(f"{nome}: MSE = {m['mse']:.8g}; PSNR = {psnr}; erro maximo = {m['erro_maximo']:.6g}")
        print(f"Qubits: {resultado['n_qubits']}; shots: {args.shots}; seed: {args.seed}")
        print(f"Amostras por pixel: {resultado['amostras'].min()} a {resultado['amostras'].max()}")
        salvar_resultados(resultado, args.saida, fonte, mostrar=not args.sem_janela)
        print(f"Resultados salvos em: {args.saida.resolve()}")
    except (OSError, ValueError) as erro:
        parser.exit(1, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
