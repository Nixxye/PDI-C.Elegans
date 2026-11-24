# Conteúdo de PDI utilizado no projeto

Explica, aula por aula, **o que foi usado**, **como funciona** e **por que entra no problema dos *C. elegans***. As páginas citadas (`pág.`) são páginas do PDF em `Estudos/PDI/Slides`, e é com elas que os docstrings do código referenciam os slides. Os slides usam MATLAB; aqui tudo está em Python (NumPy/SciPy), e a coluna "Equivalente" mostra a função do MATLAB que aparece no slide.

> Fontes: texto extraído dos PDFs e o `Resumo_Slides.md`. Fórmulas e figuras que existem só como imagem nos slides não foram conferidas; onde isso importa, a explicação abaixo vem do conhecimento geral da técnica, não do slide.

## Visão geral: por que esse conteúdo

O problema tem duas partes: **(1)** separar vermes do fundo e **(2)** separar vermes **entre si** quando se tocam. As aulas 03 a 08 resolvem (1) e dão as ferramentas para (2). A Aula 08 traz o pipeline-base do projeto (`Imagem → Top-hat → Limiarização → Morfologia → Labeling`, pág. 71). O que os slides **não** resolvem — vermes sobrepostos — é o que motiva a CNN.

| Aula | Tema | Usado em |
|---|---|---|
| 01c | vizinhança, distância | rotulagem, transformada de distância |
| 03 | transformações de intensidade | `Normalize` |
| 04 | filtros lineares | `Denoise` |
| 05 | filtro bilateral | `Denoise` (opcional) |
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

**Gamma `s = c·rᵞ` (pág. 14)** e comparação entre transformações — linear, sigmoide, piecewise, gamma (pág. 17). Cada pixel é remapeado só pelo seu próprio valor; γ < 1 clareia, γ > 1 escurece, redistribuindo o histograma (pág. 23).
- *No projeto (`ops.gamma_correct`):* realçar vermes translúcidos antes de limiarizar. Padrão γ = 1 (desligado).

**Autocontraste com saturação nos percentis (pág. 17, 23-24).** Estica o histograma entre dois percentis.
- *No projeto (`ops.autocontrast`):* opcional e **desligado por padrão**: aplicado quadro a quadro, muda o brilho do fundo e atrapalha a subtração de fundo entre quadros.
- *Equivalente:* `stretchlim` / `imadjust`.

## Aula 04 — Filtros por vizinhança

**Correlação e convolução; filtros lineares (LSI).** Cada pixel vira uma combinação ponderada da sua vizinhança por um kernel.

**Passa-baixas gaussiano (pág. 13-20).** Kernel `h = exp(−(x²+y²)/2σ²)/2πσ²`, normalizado (ganho DC = 1), lado ≈ `2·⌈3σ⌉+1`. Atenua altas frequências (ruído, granulação).
- *No projeto (`ops.gaussian`):* suavização padrão, σ = 1, antes de limiarizar. Usa `truncate=3`, que reproduz exatamente o tamanho de kernel do slide.
- *Equivalente:* `imgaussfilt`.

**Mediana (pág. 27-28).** Filtro não linear, bom para ruído sal-e-pimenta.
- *No projeto (`ops.median`):* opção em `denoise.method=median`.

## Aula 05 — Filtros adaptativos

**Filtro bilateral (pág. 7-15).** Média ponderada em que o peso é o produto de uma gaussiana **espacial** e uma gaussiana de **intensidade**: pixels vizinhos de valor muito diferente pesam pouco, então as bordas são preservadas. Não é LSI.
- *No projeto (`ops.bilateral`):* limpa o fundo sem borrar o contorno do verme. Implementado pela definição (somando versões deslocadas da imagem), por isso é lento; é opcional.
- *Equivalente:* `imbilatfilt`.

**Métricas de qualidade — MSE, PSNR, SSIM (pág. 18-22).** Comparam duas imagens com referência.
- *No projeto (`ops.mse/psnr/ssim`, `stages/timeline.py`):* **usadas na linha do tempo**: o SSIM entre as máscaras de quadros vizinhos decide se são o mesmo verme (MSE e PSNR vão para o `timeline.csv`). Para avaliar a segmentação em si continua valendo Dice/IoU (ver "Fora dos slides").

## Aula 07 — Domínio da frequência

**Filtragem homomórfica (pág. 56-64).** Separa iluminação (baixas frequências) de refletância/detalhe (altas) para corrigir iluminação desigual.
- *No projeto:* **não implementada**. É citada como alternativa ao top-hat para corrigir o fundo.

## Aula 08 — Segmentação (núcleo do projeto)

**Limiarização global e Otsu (pág. 10-13).** Escolhe o limiar que minimiza a variância dentro das duas classes (equivalente a maximizar a variância entre elas), a partir do histograma.
- *No projeto (`ops.otsu_threshold`, implementado à mão):* separa verme de fundo no `signal`. Duas adaptações práticas: um **ganho** de contraste antes (herdado da Aula 03) e um **piso** no limiar, porque em quadros sem vermes o Otsu separaria só ruído.
- *Equivalente:* `graythresh` / `imbinarize`.

**Limiarização adaptativa (pág. 14-15).** O limiar varia por região (por padrão, a média local), útil quando a iluminação não é uniforme. O slide compara Otsu × adaptativo.
- *No projeto (`ops.local_mean_threshold`):* `threshold.method=adaptive`. Como o verme é **claro** no `signal`, o limiar é `média_local + offset` (o slide trata objeto escuro).
- *Equivalente:* `adaptthresh` + `imbinarize`.

**Morfologia binária (pág. 19-35).** Operações com um elemento estruturante (SE): **dilatação** cresce o objeto (pág. 20-21), **erosão** o reduz (22-23), **abertura** = erosão + dilatação, limpa sem crescer (24-25), **fechamento** = dilatação + erosão, preenche buracos sem reduzir (26-27); comparação nas pág. 28 e 33. O tamanho do SE controla a intensidade.
- *No projeto (`ops.opening`, `ops.closing`, estágio `Morphology`):* abertura remove ruído de pixels isolados; fechamento religa um verme quebrado por reflexo; o preenchimento de buracos fecha miolos de vermes enrolados. SE em disco, raio 1.

**Esqueletização (pág. 37-48).** Reduz um objeto a uma linha fina. Duas abordagens: erosões sucessivas (pode romper conexões, pág. 38) e **afinamento** (preserva conectividade, pág. 45, `bwmorph(A,'skel',Inf)`). O slide observa que o esqueleto pode ter mais componentes que o objeto sem que isso seja erro (pág. 44).
- *No projeto (`ops.thin`, algoritmo de Zhang-Suen; `ops.skeleton_stats`):* do esqueleto saem **comprimento**, **número de pontas** (pixels com 1 vizinho) e **junções** (≥ 3 vizinhos). Juntas, indicam cruzamentos: uma junção dentro de um componente sugere vermes cruzados. Esses atributos vão para o `ComponentAnalysis`.

**Rotulagem / labeling (pág. 50-60).** Dá um rótulo a cada objeto conexo. Algoritmo em duas passadas com union-find: na 1ª, propaga-se o menor rótulo dos vizinhos já rotulados e registram-se as colisões (pág. 56); na 2ª, resolve-se cada rótulo até a raiz (pág. 60).
- *No projeto:* `ops.label_components` (rápida, SciPy, vizinhança-8) é usada no pipeline; `ops.two_pass_labeling` é a **versão do slide, escrita à mão**, usada nos testes para validar a rápida (resultados idênticos).
- *Equivalente:* `bwlabel`.

**Top-hat (pág. 62-67) e pipeline completo (pág. 71).** A limiarização global falha com iluminação não uniforme (pág. 62). O top-hat (fechamento − imagem, para objetos escuros) uniformiza o fundo (pág. 66). Pipeline do slide: `Imagem → Top-hat → Limiarização → Morfologia → Labeling`.
- *No projeto (`ops.black_tophat`, `background.method=tophat`):* alternativa à mediana temporal quando não há vários quadros ou há muito verme parado. O SE é quadrado (por velocidade) e deve ser maior que a largura do verme.

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
| **CNN** (`celegans_seg/cnn.py`) | esqueleto: a rede que separa os vermes dentro de cada amontoado será feita pelo colega | é o problema que os slides não resolvem: vermes sobrepostos |
| **Detecção de amontoado** | área ≥ 1,5 × área de referência (mediana das áreas) | decide quem segue direto e quem vai para a CNN. Subestima quando há sobreposição forte |
| **Avaliação** | Dice (união das máscaras), precisão/recall/F1 a IoU ≥ 0,5 por instância, recall nos amontoados | métricas padrão de segmentação; PSNR/SSIM (Aula 05) não medem sobreposição de máscaras |

## Onde cada técnica está no código

| Técnica | Arquivo / função |
|---|---|
| Gamma, autocontraste | `ops.gamma_correct`, `ops.autocontrast` → `Normalize` |
| Gaussiano, mediana, bilateral | `ops.gaussian/median/bilateral` → `Denoise` |
| Top-hat | `ops.black_tophat` → `BackgroundSubtraction` |
| Otsu, adaptativo | `ops.otsu_threshold`, `ops.local_mean_threshold` → `Threshold` |
| Abertura, fechamento | `ops.opening`, `ops.closing` → `Morphology` |
| Rotulagem | `ops.label_components`, `ops.two_pass_labeling` → `Labeling` |
| Esqueleto, distância | `ops.thin`, `ops.skeleton_stats`, `ops.distance_transform` → `ComponentAnalysis` |

Veja o `README.md` para o diagrama do pipeline e `docs/` para os experimentos e os dados.
