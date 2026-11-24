# Conteúdo de PDI utilizado no projeto

Explica, aula por aula, **o que foi usado**, **como funciona** e **por que entra no problema dos *C. elegans***. As páginas citadas (`pág.`) são páginas do PDF em `Estudos/PDI/Slides`, e é com elas que os docstrings do código referenciam os slides. Os slides usam MATLAB; aqui tudo está em Python (NumPy/SciPy), e a coluna "Equivalente" mostra a função do MATLAB que aparece no slide.

> Fontes: texto extraído dos PDFs e o `Resumo_Slides.md`. Fórmulas e figuras que existem só como imagem nos slides não foram conferidas; onde isso importa, a explicação abaixo vem do conhecimento geral da técnica, não do slide.

## Visão geral: por que esse conteúdo

O problema tem duas partes: **(1)** separar vermes do fundo e **(2)** separar vermes **entre si** quando se tocam. As aulas 03 a 08 resolvem (1) e dão as ferramentas para (2). A Aula 08 traz o pipeline-base do projeto (`Imagem → Top-hat → Limiarização → Morfologia → Labeling`, pág. 71). O que os slides **não** resolvem — vermes sobrepostos — é o que motiva a CNN.

| Aula | Tema | Usado em |
|---|---|---|
| 01c | vizinhança, distância | rotulagem, transformada de distância |
| 03 | transformações de intensidade | testadas, **não entram** (ver abaixo) |
| 04 | filtro gaussiano | `Denoise` |
| 05 | MSE, PSNR, SSIM | `Timeline` (liga as máscaras entre quadros) |
| 07 | domínio da frequência | só citado como alternativa |
| 08 | segmentação e morfologia | `Threshold`, `Morphology`, `Labeling`, `ComponentAnalysis`, top-hat |
| 10 | imagens coloridas | conversão para cinza, cores dos rótulos |
| 11 | watershed | não usado (testado e removido; ver `docs/EXPERIMENTOS.md`) |

---

## Aula 01c — Vizinhança e distâncias

**Vizinhança-4, diagonal e vizinhança-8 (pág. 9).** Dois pixels de objeto são "vizinhos" se se tocam; a vizinhança-8 inclui as diagonais. Isso define o que é uma região conexa.
- *No projeto:* a rotulagem usa vizinhança-8 (`ops.STRUCT_8`), para que um verme fino e diagonal não seja quebrado em pedaços.

**Distância entre pixels e `bwdist` (pág. 11).** A transformada de distância dá, para cada pixel, a distância ao pixel de fundo mais próximo. Pixels no meio de uma região grossa têm valor alto; nas bordas, valor baixo.
- *No projeto (`ops.distance_transform`):* serve para (a) medir a **espessura** de cada componente (usada também nos recortes para a CNN).
- *Equivalente:* `bwdist(~mask)`.

## Aula 03 — Transformações ponto a ponto

**Gamma `s = c·rᵞ` (pág. 14)**, comparação entre transformações — linear, sigmoide, piecewise, gamma (pág. 17) — e **autocontraste com saturação nos percentis (pág. 17, 23-24)**. Cada pixel é remapeado só pelo seu próprio valor.
- *No projeto:* **não usados.** Foram testados e ficavam desligados; aplicados quadro a quadro mudam o brilho do fundo e atrapalham a subtração de fundo entre quadros. O código foi removido (está no git). Do slide fica só a ideia de realçar o contraste antes do Otsu, feita por um simples **ganho** (`threshold.gain`, escolha nossa).

## Aula 04 — Filtros por vizinhança

**Correlação e convolução; filtros lineares (LSI).** Cada pixel vira uma combinação ponderada da sua vizinhança por um kernel.

**Passa-baixas gaussiano (pág. 13-20).** Kernel `h = exp(−(x²+y²)/2σ²)/2πσ²`, normalizado (ganho DC = 1), lado ≈ `2·⌈3σ⌉+1`. Atenua altas frequências (ruído, granulação).
- *No projeto (`ops.gaussian`):* suavização antes de limiarizar. **σ = 1 é escolha nossa.** Usa `truncate=3`, que dá o kernel do slide (pág. 18: `2·⌈3σ⌉+1`; 7×7 para σ = 1; para σ não inteiro o SciPy pode diferir em 1 pixel).
- *Equivalente:* `imgaussfilt`.

**Mediana (pág. 27-28).** Filtro não linear, bom para ruído sal-e-pimenta.
- *No projeto:* **não usada** (removida).

## Aula 05 — Filtros adaptativos

**Filtro bilateral (pág. 7-15).** Média ponderada com peso espacial × peso de intensidade (preserva bordas).
- *No projeto:* **não usado** (removido; era opcional e não entrou na configuração final).

**Métricas de qualidade — MSE (pág. 17), PSNR (pág. 18), SSIM (pág. 19-22).** Comparam duas imagens com referência. As fórmulas do slide são imagens; a implementação segue o artigo de Wang et al. (2004), que o slide cita (janela gaussiana 11×11, σ = 1,5, K1 = 0,01, K2 = 0,03).
- *No projeto (`ops.mse/psnr/ssim`, `stages/timeline.py`):* **usadas na linha do tempo**: o SSIM entre as máscaras de quadros vizinhos decide se são o mesmo verme (MSE e PSNR vão para o `timeline.csv`). Como o fundo vazio das máscaras é igual nas duas imagens e daria SSIM = 1, a média é feita só em volta das máscaras (ESCOLHA NOSSA). Para avaliar a segmentação em si continua valendo Dice/IoU (ver "Fora dos slides").

## Aula 07 — Domínio da frequência

**Filtragem homomórfica (pág. 56-64).** Separa iluminação (baixas frequências) de refletância/detalhe (altas) para corrigir iluminação desigual.
- *No projeto:* **não implementada**. É citada como alternativa ao top-hat para corrigir o fundo.

## Aula 08 — Segmentação (núcleo do projeto)

**Limiarização global e Otsu (pág. 10-13).** Escolhe o limiar que minimiza a variância dentro das duas classes (equivalente a maximizar a variância entre elas), a partir do histograma.
- *No projeto (`ops.otsu_threshold`, implementado à mão):* separa verme de fundo no `signal`. Duas adaptações **nossas**: um **ganho** de contraste antes (`threshold.gain = 4`) e um **piso** no limiar (`threshold.min_threshold = 40`), porque em quadros sem vermes o Otsu separaria só ruído.
- *Equivalente:* `graythresh` / `imbinarize`.

**Limiarização adaptativa (pág. 14-15, 68).** O limiar varia por região (a média local), útil quando a iluminação não é uniforme.
- *No projeto:* **não usada** (removida). A versão que tínhamos usava `média_local + offset`, diferente da regra do slide (pág. 68: pixel `t` % abaixo da média local), e nunca foi a configuração final.

**Morfologia binária (pág. 19-35).** Operações com um elemento estruturante (SE): **dilatação** cresce o objeto (pág. 20-21), **erosão** o reduz (22-23), **abertura** = erosão + dilatação, limpa sem crescer (24-25), **fechamento** = dilatação + erosão, preenche buracos sem reduzir (26-27); comparação nas pág. 28 e 33. O tamanho do SE controla a intensidade.
- *No projeto (`ops.opening`, `ops.closing`, estágio `Morphology`):* abertura remove ruído de pixels isolados; fechamento religa um verme quebrado por reflexo; o preenchimento de buracos fecha miolos de vermes enrolados. SE em disco, raio 1 (**escolha nossa**, `morphology.opening_radius`/`closing_radius`).

**Esqueletização (pág. 37-48; a pág. 48 está marcada "PULAR" no slide anotado).** Reduz um objeto a uma linha fina. Duas abordagens: erosões sucessivas (pode romper conexões, pág. 38) e **afinamento** (preserva conectividade, pág. 45, `bwmorph(A,'skel',Inf)`). O slide observa que o esqueleto pode ter mais componentes que o objeto sem que isso seja erro (pág. 44).
- *No projeto (`ops.thin`, algoritmo de Zhang-Suen; `ops.skeleton_stats`):* do esqueleto saem **comprimento**, **número de pontas** (pixels com 1 vizinho) e **junções** (≥ 3 vizinhos). Juntas, indicam cruzamentos: uma junção dentro de um componente sugere vermes cruzados. Esses atributos vão para o `ComponentAnalysis`.

**Rotulagem / labeling (pág. 50-60).** Dá um rótulo a cada objeto conexo. Algoritmo em duas passadas com union-find: na 1ª, propaga-se o menor rótulo dos vizinhos já rotulados e registram-se as colisões (pág. 56); na 2ª, resolve-se cada rótulo até a raiz (pág. 60).
- *No projeto:* `ops.label_components` (rápida, SciPy, vizinhança-8) é usada no pipeline; `ops.two_pass_labeling` é a **versão do slide, escrita à mão**, usada nos testes para validar a rápida (resultados idênticos).
- *Equivalente:* `bwlabel`.

**Top-hat (pág. 62-67) e pipeline completo (pág. 71).** A limiarização global falha com iluminação não uniforme (pág. 62). A transformada top-hat do slide é `TopHat = G − (G ∘ S)`, com `∘` a abertura em tons de cinza (pág. 66), e realça objetos **claros**. Pipeline do slide: `Imagem → Top-hat → Limiarização → Morfologia → Labeling`.
- *No projeto (`ops.tophat`, `ops.black_tophat`):* como o verme é **escuro**, aplicamos a transformada do slide à imagem invertida (`1 − G`), o que equivale a fechamento − imagem. Entra em `signal_spatial`, para completar a cauda e achar o verme parado (`threshold.grow_*`, `spatial_min_area`, **fora dos slides**). O SE é um quadrado de 31 px (escolha nossa, por velocidade; o slide usa SE circular, pág. 65).

## Aula 10 — Imagens coloridas

**RGB → cinza: `0,299·R + 0,587·G + 0,114·B` (pág. 2).**
- *No projeto (`io.to_gray_u8`):* qualquer entrada colorida é convertida para cinza com essa fórmula (a mesma do modo `'L'` do PIL).

**Pseudocolorização de rótulos (pág. 29, `label2rgb`).**
- *No projeto (`stages/viz.py`):* cada verme (instância) ganha uma cor distinta nos contornos e nos painéis.

## Aula 11 — Watershed

Testado para separar amontoados (distância + watershed, F1 0,69 nos recortes de teste) e **removido** do código: vermes cruzados ou colados em paralelo não têm "pescoço". Detalhes em `docs/EXPERIMENTOS.md`; o código está no git (commit `3ecc77f`).

---

## Fora dos slides (o que foi acrescentado)

| Elemento | O quê | Por quê |
|---|---|---|
| **Fundo por mediana temporal** | mediana de ~25 quadros espalhados na sequência; `signal = fundo − quadro` | vermes se movem, a placa não: a mediana "apaga" os vermes e sobra só o fundo. Limite: verme parado entra no fundo |
| **Ajustes para verme parado** | percentil 70 (e não a mediana) no fundo; `labeling.bridge_gap` liga pedaços de um verme partido; `threshold.grow_dist` completa a cauda com o top-hat | o fundo temporal absorve a parte do verme que fica parada (ver `docs/EXPERIMENTOS.md`) |
| **CNN** (`celegans_seg/cnn.py`, `cnn/`) | Mask R-CNN do colega que separa os vermes dentro de cada amontoado que o PDI escolhe | é o problema que os slides não resolvem: vermes sobrepostos |
| **Detecção de amontoado** | área ≥ 1,5 × área de referência (mediana das áreas) | decide quem segue direto e quem vai para a CNN. Subestima quando há sobreposição forte |
| **Linha do tempo** (`stages/timeline.py`) | o SSIM (Aula 05) entre as máscaras decide se dois vermes de quadros vizinhos são o mesmo; a ligação (atribuição húngara, distância máxima, tolerância de 2 quadros) é nossa | seguir cada verme no vídeo; os slides dão a medida de similaridade, não o rastreamento |
| **Avaliação** | IoU, Dice, precisão/recall/F1 por instância, pareamento húngaro (ver "Métricas" abaixo) | métricas padrão de segmentação; MSE/PSNR/SSIM (Aula 05) não dizem se cada verme foi encontrado |

## Métricas

Há **dois grupos**: as que os slides ensinam (Aula 05) e as que usamos para avaliar a segmentação (que **não** estão nos slides; procurei Dice, IoU, Jaccard, F1, precisão e recall nos 19 PDFs e não achei).

### Métricas dos slides: MSE, PSNR, SSIM (Aula 05, pág. 17-22)

São métricas de **qualidade de imagem com referência**: comparam uma imagem `a` com uma imagem de referência `b` do mesmo tamanho. As fórmulas dos slides são imagem (não saem no texto extraído); abaixo, as definições padrão, que é o que o código implementa.

| Métrica | Fórmula | Como ler |
|---|---|---|
| **MSE** (pág. 17) | `(1/N) Σ (a − b)²` | média do erro ao quadrado, pixel a pixel. 0 = idênticas. Quanto menor, melhor |
| **PSNR** (pág. 18) | `10 · log₁₀(L² / MSE)` dB, `L` = valor máximo (1 em [0,1]) | MSE em escala logarítmica. Quanto **maior**, melhor; infinito se forem idênticas |
| **SSIM** (pág. 19-22) | `(2μₐμ_b + C₁)(2σₐ_b + C₂) / ((μₐ² + μ_b² + C₁)(σₐ² + σ_b² + C₂))` | compara **luminância, contraste e estrutura** numa janela gaussiana (11×11, σ = 1,5, `C₁ = (0,01 L)²`, `C₂ = (0,03 L)²`) e tira a média. Vai de −1 a 1; **1 = idênticas** |

- *Onde usamos (`ops.mse/psnr/ssim`):* na **linha do tempo**, para decidir se duas máscaras de quadros vizinhos são o mesmo verme (SSIM alto = mesma forma). **Não** servem para avaliar a segmentação do PDI.
- *Cuidado com o SSIM em máscaras:* o fundo vazio é igual nas duas imagens e daria SSIM ≈ 1, escondendo a diferença; por isso a média é feita só **em volta das máscaras** (escolha nossa).
- *Por que MSE/PSNR não avaliam a segmentação:* o quadro tem 912 × 736 = 671.232 pixels e um verme ocupa ~580. Se a segmentação **perder um verme inteiro**, o MSE entre as máscaras é `580 / 671.232 ≈ 0,00086`, ou seja, **PSNR ≈ 30,6 dB**, que "parece ótimo". O fundo domina a conta. Por isso precisamos de uma métrica que olhe **cada verme** (IoU, F1).

### Métricas que usamos para a segmentação (fora dos slides)

Todas comparam as máscaras previstas com o **gabarito** (uma máscara por verme). `P` = máscara prevista, `G` = máscara do gabarito.

| Métrica | Definição | Como ler |
|---|---|---|
| **IoU** (interseção sobre união) | `\|P ∩ G\| / \|P ∪ G\|` | 1 = máscara igual à do gabarito; 0 = nenhuma sobreposição. **0,5 é o limite de "acerto"**: por exemplo, uma máscara que cobre o verme inteiro **e** mais uma área igual a ele dá IoU 0,5 |
| **Dice** | `2 \|P ∩ G\| / (\|P\| + \|G\|)` | parecida com o IoU, sempre maior ou igual (`Dice = 2·IoU / (1 + IoU)`). Usamos o **Dice da união** de todas as máscaras previstas contra a união do gabarito (`dice_semantic`): mede se "o que é verme" foi pintado, sem olhar quem é quem |
| **Pareamento húngaro** | atribuição 1 para 1 entre vermes previstos e do gabarito que maximiza o IoU total | evita contar duas vezes o mesmo verme e dá um **TP** (verdadeiro positivo) quando o par tem IoU ≥ 0,5 |
| **Precisão** | `TP / (TP + FP)` | dos vermes que o programa marcou, quantos são de verdade. `FP` = marcou e não é verme (ou não casou) |
| **Recall** | `TP / (TP + FN)` | dos vermes do gabarito, quantos o programa achou. `FN` = verme perdido |
| **F1** | `2 · precisão · recall / (precisão + recall)` | um número só que só é alto se precisão **e** recall forem altos. `F1@0,75` = o mesmo com IoU ≥ 0,75 (exige contorno mais fiel) |

Métricas derivadas, usadas para olhar os amontoados sem depender de separá-los: `f1_single` (F1 só dos vermes isolados), `cluster_dice` (Dice da máscara de amontoados prevista × a do gabarito, onde amontoado = componente conexo do gabarito com ≥ 2 vermes) e `cluster_recall` (dos vermes que estão em amontoados, quantos foram recuperados).

No relatório do PDI (`tools/pdi_report.py`) há três taxas simples, com limites configuráveis: **verme encontrado** (pelo menos metade do verme cai dentro da máscara, `--found-thr 0,5`), **acerto da máscara** de um verme isolado (IoU ≥ 0,5, `--hit-iou`) e **falsos positivos** (componente com menos de 30 % da área sobre algum verme do gabarito, depois de descontar bordas e vermes não anotados; ver `docs/EXPERIMENTOS.md`).

### Cuidados para não apresentar número errado

- **Gabarito duplicado:** duas máscaras quase iguais para o mesmo verme contam como um verme não encontrado, e o **recall parece pior**. Em `24_1_1_1` e `24_1_1_2` quase metade são duplicatas. O `pdi_report.py` funde antes de medir.
- **Gabarito incompleto:** o CSB-1 não anota vermes cortados pela borda nem alguns vermes inteiros. O programa os marca, e eles contam como **falso positivo**, o que **derruba a precisão** sem ser erro dele.
- **Um amontoado conta como um verme só** no PDI: o F1 por instância do PDI sozinho (≈ 0,74 na sequência de teste) é bem menor que a taxa de acerto dos vermes isolados (99,9 %). São medidas diferentes; diga qual está mostrando.

## Onde cada técnica está no código

| Técnica | Arquivo / função |
|---|---|
| Gaussiano | `ops.gaussian` → `Denoise` |
| Top-hat | `ops.tophat`, `ops.black_tophat` → `BackgroundSubtraction` |
| Otsu | `ops.otsu_threshold` → `Threshold` |
| Abertura, fechamento | `ops.opening`, `ops.closing` → `Morphology` |
| Rotulagem | `ops.label_components`, `ops.two_pass_labeling` → `Labeling` |
| Esqueleto, distância | `ops.thin`, `ops.skeleton_stats`, `ops.distance_transform` → `ComponentAnalysis` |
| MSE, PSNR, SSIM | `ops.mse`, `ops.psnr`, `ops.ssim` → `Timeline` |
| IoU, Dice, precisão, recall, F1, pareamento húngaro | `stages/evaluate.py` (`iou_matrix`, `Evaluate`), `cnn.match_scores` |

Veja o `README.md` para o diagrama do pipeline e `docs/` para os experimentos e os dados.
