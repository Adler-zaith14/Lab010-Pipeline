# Laboratório 10 — O Pipeline Definitivo (RAG, QLoRA e Otimização de Inferência na GPU)

**Instituição:** ICEV — Instituto de Ensino Superior  
**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Dimmy Magalhães  
**Autor:** Adler Castro Alves

---

## Nota de Integridade

Este laboratório foi desenvolvido de forma individual, com base nos materiais disponibilizados em aula e nas documentações oficiais das bibliotecas utilizadas. Os erros encontrados durante a execução foram identificados e corrigidos manualmente. Durante o desenvolvimento, identifiquei que o parâmetro `attn_implementation="sdpa"` não era a configuração correta, então tive que mudar para `attn_implementation="flash_attention_2"`. Também corrigi outros parâmetros e pedi auxílio à IA para organizar o versionamento. Todo esse processo de depuração e correção foi realizado por mim ao longo da execução do laboratório.


---

## Métricas de Benchmark

| Métrica | Sem Cache | Otimizado (KV Cache + FlashAttention-2) |
|---|---|---|
| VRAM na carga do modelo (QLoRA 4-bit) | 789 MB | — |
| Tokens no contexto RAG simulado | 2048 | 2048 |
| Tempo de geração (s) | 11.42 | 6.46 |
| Pico VRAM durante geração (MB) | 2110.75 | 2110.16 |
| Speedup | — | 1.77x |
| Variação de VRAM | — | 0.6% |

---

## Configuração do Ambiente

- **GPU:** NVIDIA Tesla T4 (16 GB VRAM)
- **Modelo:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Quantização:** QLoRA 4-bit via `bitsandbytes` (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`, `bnb_4bit_quant_type="nf4"`)
- **Otimização de hardware:** FlashAttention-2 (`attn_implementation="flash_attention_2"`)
- **Otimização de software:** KV Cache (`use_cache=True`)

---

## Análise Arquitetural (Passo 5)

### Parte A — Como QLoRA, KV Cache e FlashAttention-2 salvaram o pipeline

O principal gargalo de memória em Transformers generativos vem de três frentes simultâneas: o tamanho dos pesos do modelo, a complexidade quadrática O(n²) do Self-Attention sobre sequências longas e o recálculo redundante das matrizes de chave e valor (K, V) a cada novo token gerado. Neste laboratório, cada uma dessas frentes foi atacada com uma técnica específica. O QLoRA reduziu a pegada dos pesos de ~2,2 GB (FP16 completo) para aproximadamente 789 MB, ao representar os parâmetros em 4 bits com tipo NF4 e executar os cálculos em FP16 apenas quando necessário — isso garantiu que o modelo coubesse na VRAM antes mesmo de processar qualquer token. O KV Cache eliminou o trabalho redundante do decodificador: em vez de recalcular as ativações K e V de todos os tokens do contexto a cada passo de geração, o modelo armazena esses vetores em memória e apenas processa o novo token gerado, reduzindo a latência de forma proporcional ao tamanho do contexto. Finalmente, o FlashAttention-2 reorganiza o cálculo da atenção para operar inteiramente dentro da memória SRAM de alta velocidade da GPU, evitando a leitura e escrita repetidas na HBM (memória global), que é ordens de magnitude mais lenta — o que reduz o pico de VRAM durante a fase de prompting e acelera o tempo total de geração mesmo sem alterar a precisão matemática do resultado.

### Parte B — Por que o FlashAttention-2 falharia com 2 milhões de tokens

O FlashAttention-2 resolve o problema de *bandwidth*, não o problema de *capacidade*. Mesmo reorganizando os acessos à memória para trabalhar em blocos na SRAM, ele ainda precisa materializar o KV Cache inteiro na VRAM para toda a sequência de entrada — e esse cache cresce linearmente com o número de tokens: `O(n * d_model * n_layers)`. Para 2 milhões de tokens com um modelo de porte médio (32 camadas, d_model = 4096), apenas o KV Cache ocuparia dezenas de gigabytes, tornando inviável o uso em qualquer GPU de consumo ou mesmo em instâncias cloud de custo acessível. Além disso, a própria atenção full (mesmo com FlashAttention-2) exige que cada token atenda a todos os outros, o que, a 2 milhões de tokens, resulta em 4 × 10¹² operações por camada — computacionalmente proibitivo. É precisamente por esse motivo que a indústria está migrando para arquiteturas de **State Space Models (SSMs)** como o **Mamba**: em vez de manter um cache de toda a sequência vista até agora, o Mamba comprime o histórico em um vetor de estado oculto de tamanho fixo que é atualizado recorrentemente a cada novo token. Isso garante complexidade de memória **O(1)** em relação ao comprimento da sequência — o estado tem sempre o mesmo tamanho, independentemente de a sequência ter 1.000 ou 2.000.000 tokens — tornando o processamento de contextos ultra-longos viável sem explosão de VRAM.

---

## Notas sobre o Ambiente

### Sobre o FlashAttention-2

O FlashAttention-2 exige compilação de extensões CUDA nativas, instalado com:

```bash
!pip install flash-attn --no-build-isolation
