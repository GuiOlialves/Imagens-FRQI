# FRQI v2 - codificação e reconstrução de imagem

A v2 implementa o objetivo: imagem clássica -> estado quântico FRQI -> medições -> imagem clássica reconstruída. É um experimento simulado com Qiskit Aer.

`frqi.py` e `README.md` continuam como a versão original, sem alterações. Execute `frqi_v2.py` para usar as correções; os resultados novos ficam em `resultados_v2/`.

## Correções

1. **Ângulo da intensidade:** para `theta = pixel * pi/2`, o circuito aplica `mcry(2 * theta, ...)`. A porta RY divide o ângulo por dois nas amplitudes; assim o estado de cor é `cos(theta)|0> + sin(theta)|1>` e a probabilidade de cor 1 é `sin(theta)^2`.
2. **Posições dos pixels:** o bit menos significativo do índice vai para o primeiro qubit de posição. Na medição, o último bit é a cor e os demais formam o mesmo índice usado na codificação. A reconstrução mantém a ordem dos pixels.
3. **Comparação ideal e medida:** a imagem ideal é calculada do estado produzido pelo circuito. A imagem medida vem das contagens do Aer. A primeira permite verificar codificação e reconstrução; a segunda acrescenta a incerteza de amostragem.
4. **Pixels não observados:** recebem `NaN`, aparecem em magenta no painel e ficam em branco no CSV. Não são tratados como pixels pretos. MSE e PSNR globais são reportados apenas se todas as posições forem observadas.
5. **Execução reproduzível:** há sementes para o simulador e a transpilação, seleção explícita de shots e dados exportados com métricas e versões das bibliotecas.

O experimento da v2 foca na reconstrução. O circuito RZ/CNOT da v1 permanece no arquivo original. Nessa sequência, RZ modifica fases e CNOT permuta os estados de posição; as medições na base computacional não tornam a chave RZ visível nas intensidades. Reversibilidade de um circuito não demonstra segurança criptográfica.

## Instalação

Python 3.10 ou superior:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements_v2.txt
```

Em macOS/Linux, use `.venv/bin/python` no lugar de `.venv\Scripts\python`.
TensorFlow não é necessário na v2: o mesmo dataset MNIST é lido diretamente do arquivo NPZ oficial.

## Executar

Uma demonstração sem download, com todos os níveis de cinza e posições assimétricas:

```powershell
.venv\Scripts\python frqi_v2.py --demo --sem-janela
```

O mesmo exemplo da versão original (MNIST, índice 15, 8 x 8, 6.400 shots):

```powershell
.venv\Scripts\python frqi_v2.py --mnist-indice 15 --sem-janela
```

Sem argumentos, o programa também usa o MNIST no índice 15. Na primeira execução, baixa `mnist.npz` da URL oficial do TensorFlow para `dados/`; nas seguintes, reutiliza o arquivo. Para usar uma cópia existente sem download:

```powershell
.venv\Scripts\python frqi_v2.py --mnist-indice 15 --mnist-arquivo "caminho/mnist.npz" --sem-janela
```

Uma imagem sua, convertida para escala de cinza e redimensionada:

```powershell
.venv\Scripts\python frqi_v2.py --imagem "minha_imagem.png" --shots 64000 --seed 42 --sem-janela
```

Outras opções: `--lado 2|4|8|16`, `--saida pasta` e `--seed numero`. Remova `--sem-janela` para exibir o painel interativamente.

O padrão permanece em 6.400 shots para comparação com a v1. Em 8 x 8, isso corresponde a aproximadamente 100 amostras por posição. Mais shots tendem a diminuir o erro estatístico; uma execução individual não garante melhora monotônica.

## O que é reconstruído

A referência é a imagem **depois** de converter para cinza e redimensionar para 8 x 8 (ou o lado selecionado). Redimensionar um MNIST de 28 x 28 para 8 x 8 perde detalhes; esta v2 não promete recuperar os 28 x 28 originais.

O circuito codifica, para N pixels:

```text
|I> = 1/sqrt(N) * sum_i [cos(theta_i)|0> + sin(theta_i)|1>]cor |i>posicao
theta_i = pixel_i * pi/2
P(cor=1 | posicao=i) = sin(theta_i)^2
pixel_i = 2/pi * arcsin(sqrt(P(cor=1 | posicao=i)))
```

São `log2(N) + 1` qubits. Um shot devolve uma posição e um bit de cor; são necessárias várias preparações e medições do circuito para estimar todos os pixels. O simulador é ideal, sem modelo de ruído de hardware.

A referência ideal usa o vetor de estado do simulador, não copia a matriz original. Esse acesso completo ao estado não está disponível em hardware quântico. A comparação permite separar erros sistemáticos na implementação do erro das medições finitas; não demonstra vantagem de desempenho sobre processamento clássico.

## Resultados

Todos são escritos em `resultados_v2/`, ou na pasta indicada por `--saida`:

- `frqi_v2_comparacao.png`: imagem de referência, FRQI ideal, imagem reconstruída das medições e mapa de erro absoluto. As três imagens usam a mesma escala de intensidade, de 0 a 1.
- `frqi_v2_imagens.npz`: matrizes `original`, `ideal`, `reconstruida` e `amostras_por_pixel`.
- `frqi_v2_metricas.json`: MSE, PSNR, cobertura de pixels, shots, seed, tempos, versões de bibliotecas e contagens brutas. Para MSE zero, `psnr_infinito` é `true` e `psnr_db` é `null`; para cobertura incompleta, as métricas globais são `null`.
- `frqi_v2_pixels.csv`: comparação numérica e número de amostras de cada pixel.

Uma nova execução na mesma pasta substitui apenas esses resultados da v2. Use `--saida` diferente para guardar cada rodada.

## Validação

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

Os testes verificam o estado FRQI contra sua equação teórica, a intensidade de preto/branco e tons intermediários, a ordem de uma imagem assimétrica, a leitura real das medições no Aer e o tratamento de pixels não observados.

### Execução conferida

Os oito testes passaram. Também foram executados o exemplo MNIST, a grade de demonstração e um caso com apenas um shot, no qual os pixels não observados permaneceram desconhecidos e as métricas globais foram omitidas.

Para o MNIST de índice 15 (dígito 7), com o mesmo pré-processamento da v1 e referência 8 x 8:

| Reconstrução | MSE | PSNR |
| --- | ---: | ---: |
| v1, probabilidades exatas do circuito original sem cifra | 0,0644897 | 11,91 dB |
| v2, referência ideal do simulador | 3,04 x 10^-30 | 295,17 dB |
| v2, 6.400 shots e seed 42 | 0,000509034 | 32,93 dB |

O erro máximo ideal da v2 foi 4,11 x 10^-15, da ordem da precisão numérica. O PSNR ideal muito alto corresponde a essa precisão do simulador, não a um resultado de medição física. A comparação da v1 usou seu próprio codificador e reconstrutor sobre probabilidades exatas; seu erro já existe antes da incerteza dos shots. Os arquivos originais usados correspondem à branch `main` no commit `ffe4c9b18d125abc6d1135c9d4a33a2cdcedf6de`.

## Referências

- [Qiskit RY: amplitudes com theta/2](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYGate)
- [Qiskit: ordem dos bits](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering)
- [Qiskit Statevector: referência exata do simulador](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)
- [RZ: fases na base computacional](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZGate)
- [CX: transformação dos estados de base](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.CXGate)
- [MNIST NPZ usado pelo Keras](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
