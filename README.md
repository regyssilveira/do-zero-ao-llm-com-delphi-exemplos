# Do Zero ao LLM com Delphi — exemplos

O código deste diretório acompanha o livro **Do Zero ao LLM com Delphi**.

Este é o repositório público dos exemplos. O projeto editorial e o manuscrito são mantidos separadamente.

## Plataforma oficial

- Delphi 13 Florence;
- alvo Windows 64-bit;
- build de console;
- CPU, sem biblioteca externa de machine learning.

## Licença

O código-fonte, sua documentação e os corpora autorais em `datasets/` são disponibilizados sob a Apache License 2.0. Consulte `LICENSE`, `NOTICE` e `datasets/DATASET_CARD.md`.

Datasets externos não herdam automaticamente essa licença. Todo corpus adicional deverá trazer proveniência, hash e licença próprios.

## Corpora distribuídos

- `corpus-mini.txt`: leitura rápida para os primeiros exercícios;
- `corpus-base.txt`: treinamento e benchmark da configuração-base;
- `DATASET_CARD.md`: autoria, finalidade, método de geração e limitações;
- `SHA256SUMS`: integridade das versões congeladas.

Para reproduzir e auditar os arquivos, execute `python tools/generate_corpus.py` e `python tools/audit_corpus.py` a partir da raiz deste repositório.

## Estado

O projeto `DelphiLM` implementa o percurso técnico completo do livro: configuração, tokenização, tensores, camadas, perdas, backward, otimização, embeddings, atenção causal, bloco Transformer, modelo integrado, treinamento mínimo, geração e checkpoint.

## Primeira execução

Abra `DelphiLM/DelphiLM.groupproj` no Delphi 13 Florence. O grupo contém:

- `DelphiLM.Smoke`: confirma configuração e execução Win64;
- `DelphiLM.Tests`: executa os testes automatizados;
- `DelphiLM.Benchmark`: mede loss, tempo, throughput e pico de memória no corpus-base.

A versão consolidada foi compilada no Delphi 13 Florence, compilador Win64 37.0, em Debug e Release. Execute `DelphiLM.Tests`; o critério é concluir 35 testes sem linhas de falha e encerrar com código zero. `DelphiLM.Smoke` confirma a configuração de referência e a sequência pseudoaleatória reproduzível.

O benchmark de referência usa Release/Win64. Após compilar o grupo, execute `DelphiLM.Benchmark.exe datasets\corpus-base.txt 50`. O relatório medido fica em `benchmarks/2026-08-24-win64-release.md`.
