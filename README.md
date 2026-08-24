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

O projeto `DelphiLM` começa como spike técnico. Sua finalidade inicial é validar compilação, testes, determinismo, operações numéricas, treinamento, persistência e desempenho antes da redação definitiva dos capítulos.

## Primeira execução

Abra `DelphiLM/DelphiLM.groupproj` no Delphi 13 Florence. O grupo contém:

- `DelphiLM.Smoke`: confirma configuração e execução Win64;
- `DelphiLM.Tests`: executa os testes automatizados iniciais.

O estado inicial foi compilado no Delphi 13/compilador Win64 37.0 em Debug e Release. Os testes verificam a configuração de referência, a compatibilidade entre embedding e cabeças e a repetibilidade do gerador pseudoaleatório.
