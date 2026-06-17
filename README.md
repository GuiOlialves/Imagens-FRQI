# 🔐 FRQI Quantum Image Encryption

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-Aer-6929C4?style=flat&logo=qiskit&logoColor=white)
![Status](https://img.shields.io/badge/status-pesquisa%20em%20andamento-yellow)

> Codificação de imagens em estados quânticos via **FRQI** (Flexible Representation of Quantum Images), com uma camada de **criptografia quântica** construída sobre o circuito, simulada com Qiskit Aer.

---

## 📑 Sumário

- [Sobre o projeto](#-sobre-o-projeto)
- [Contexto da pesquisa: clássico vs quântico](#-contexto-da-pesquisa-clássico-vs-quântico)
- [Como funciona](#-como-funciona)
- [Tecnologias](#-tecnologias)
- [Instalação](#-instalação)
- [Como executar](#-como-executar)
- [Resultados](#-resultados)
- [Próximos passos](#-próximos-passos)
- [Autor](#-autor)

---

## 📌 Sobre o projeto

Este script implementa o protocolo **FRQI** para representar uma imagem em escala de cinza como um estado quântico, e em seguida aplica uma **criptografia quântica customizada** sobre essa representação — cifrando e decifrando a imagem diretamente no espaço de estados, sem nunca reconstruir os pixels em texto claro durante o processo.

O pipeline completo cobre: carregamento e pré-processamento da imagem (dataset MNIST), codificação quântica, criptografia, descriptografia, simulação e reconstrução da imagem a partir das medições, com métricas de fidelidade (MSE e PSNR).

## 📐 Contexto da pesquisa: clássico vs quântico

Este projeto faz parte de uma Iniciação Científica que compara abordagens **clássicas e quânticas de processamento de imagens**. Métodos clássicos de criptografia de imagem operam diretamente sobre matrizes de pixels (permutação, XOR, transformações no domínio de frequência etc.) e sua segurança depende de propriedades computacionais bem conhecidas.

A abordagem quântica explorada aqui parte de um princípio diferente: a imagem é representada como **superposição de estados** (cada posição de pixel associada a uma amplitude de cor), e a "cifra" é aplicada como uma sequência de rotações (`RZ`) e emaranhamento (`CNOT`) sobre os qubits de posição. A informação só é recuperável corretamente se a chave usada na descriptografia for a inversa exata da transformação aplicada — qualquer chave incorreta gera uma reconstrução estatisticamente distorcida. Isso cria uma superfície de segurança fundamentalmente diferente da criptografia clássica, e é exatamente esse contraste que a pesquisa busca caracterizar.

## ⚙️ Como funciona

1. **Pré-processamento** — uma imagem do dataset MNIST é normalizada (0 a 1) e redimensionada para 8×8 pixels (64 valores), o tamanho viável para simulação com poucos qubits.

2. **Codificação FRQI** — um circuito com 1 qubit de cor + 6 qubits de posição (cobrindo as 64 posições) é construído. Os qubits de posição entram em superposição uniforme (`H`), e para cada pixel é aplicada uma rotação controlada (`MCRY`) no qubit de cor, com ângulo proporcional à intensidade do pixel (`θ = intensidade × π/2`).

3. **Criptografia quântica** — sobre os qubits de posição, aplica-se uma sequência de rotações `RZ` parametrizadas por uma **chave de 6 elementos**, seguida de uma cadeia de portas `CNOT` para emaranhar os qubits, e mais uma rodada de `RZ` com a chave escalada. Isso transforma a distribuição de probabilidades original em algo irreconhecível sem a chave.

4. **Descriptografia** — aplica-se o circuito inverso exato da criptografia, restaurando (estatisticamente) a distribuição original.

5. **Simulação** — os três circuitos (original, criptografado, recuperado) são executados no `AerSimulator` com 6400 shots cada.

6. **Reconstrução da imagem** — a partir das contagens de medição, a probabilidade de cada pixel estar no estado "1" é convertida de volta em ângulo (`θ = arcsin(√p)`) e depois em intensidade de pixel (`θ / (π/2)`).

7. **Métricas** — MSE e PSNR comparam a imagem original, a reconstruída (sem cifra) e a recuperada (após cifrar e decifrar), validando que o ciclo de criptografia/descriptografia preserva a fidelidade da imagem.

8. **Visualização** — um painel com 4 imagens lado a lado (original 8×8, reconstrução FRQI, versão criptografada, versão recuperada) é salvo como `frqi_angulos.png`.

## 🛠 Tecnologias

- **Python 3.10+**
- **Qiskit** — construção dos circuitos quânticos
- **Qiskit Aer** — simulação dos circuitos
- **NumPy** — operações numéricas e reconstrução da imagem
- **Matplotlib** — visualização dos resultados
- **scikit-image** — redimensionamento da imagem
- **TensorFlow** — carregamento do dataset MNIST

## 📥 Instalação

```bash
pip install numpy matplotlib scikit-image tensorflow qiskit qiskit-aer
```

> Recomendado usar um ambiente virtual antes de instalar:
> ```bash
> python -m venv venv
> source venv/bin/activate  # Windows: venv\Scripts\activate
> pip install numpy matplotlib scikit-image tensorflow qiskit qiskit-aer
> ```

## ▶️ Como executar

```bash
python frqi_encryption.py
```

O script imprime no terminal o dígito processado, os tempos de execução de cada simulação e as métricas de MSE/PSNR, além de salvar a imagem comparativa `frqi_angulos.png` no diretório atual.

## 📊 Resultados

Ao final da execução, o script gera:

- **Console**: dígito da imagem processada, tempos de simulação dos 3 circuitos (original, criptografado, recuperado) e métricas de MSE/PSNR comparando a imagem original com a reconstruída e com a recuperada após o ciclo completo de criptografia.
- **Imagem `frqi_angulos.png`**: painel visual com 4 versões da imagem — original, reconstrução FRQI direta, versão criptografada (irreconhecível) e versão recuperada após descriptografia — demonstrando visualmente que a chave correta restaura a imagem original.

<!-- Depois de rodar localmente, adicione aqui a imagem real gerada: -->
<!-- ![Resultado FRQI](frqi_angulos.png) -->

## 🚀 Próximos passos

- Expandir os testes para outros datasets (ex: EuroSAT)
- Modelar ruído (ex: canal depolarizante) para avaliar robustez da criptografia em hardware real
- Investigar aplicações em redes neurais quânticas a partir da representação FRQI

## 👤 Autor

**Gui** — Estudante de Sistemas de Informação (Senac Santo Amaro), pesquisa de Iniciação Científica em processamento clássico e quântico de imagens.
