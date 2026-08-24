# Do Zero ao LLM com Delphi — exemplos

O código deste diretório acompanha o livro **Do Zero ao LLM com Delphi**.

Este é o repositório público dos exemplos. O projeto editorial e o manuscrito são mantidos separadamente.

## Plataforma oficial

- Delphi 13 Florence;
- alvo Windows 64-bit;
- build de console;
- CPU, sem biblioteca externa de machine learning.

## Licença

O código-fonte e sua documentação neste diretório são disponibilizados sob a Apache License 2.0. Consulte `LICENSE` e `NOTICE`.

Datasets não herdam essa licença. Cada corpus deverá trazer proveniência, hash e licença próprios.

## Estado

O projeto `DelphiLM` implementa o percurso técnico completo do livro: configuração, tokenização, tensores, camadas, perdas, backward, otimização, embeddings, atenção causal, bloco Transformer, modelo integrado, treinamento mínimo, geração e checkpoint.

## Primeira execução

Abra `DelphiLM/DelphiLM.groupproj` no Delphi 13 Florence. O grupo contém:

- `DelphiLM.Smoke`: confirma configuração e execução Win64;
- `DelphiLM.Tests`: executa os testes automatizados iniciais.

A versão consolidada foi compilada no Delphi 13 Florence, compilador Win64 37.0, em Debug e Release. Execute `DelphiLM.Tests`; o critério é concluir 34 testes sem linhas de falha e encerrar com código zero. `DelphiLM.Smoke` confirma a configuração de referência e a sequência pseudoaleatória reproduzível.
