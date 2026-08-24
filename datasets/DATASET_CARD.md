# Dataset Card — DelphiLM Corpus Didático PT-BR

## Identificação

- Nome: DelphiLM Corpus Didático PT-BR
- Versão: 1.0.0
- Data: 24 de agosto de 2026
- Responsável editorial: Régys Borges da Silveira
- Idioma: português brasileiro
- Codificação: UTF-8 sem BOM
- Normalização: Unicode NFC
- Licença: Apache License 2.0

## Finalidade

Oferecer um corpus pequeno, redistribuível e reproduzível para demonstrar tokenização por caracteres, criação de batches, treinamento de um modelo causal, checkpoint, geração e benchmark em CPU/Win64.

Não é um dataset destinado a treinar sistemas de produção, avaliar conhecimento geral ou representar a diversidade da língua portuguesa.

## Composição

O conteúdo foi produzido especificamente para o projeto por um gerador determinístico baseado em fragmentos autorais. Ele combina:

- pequenas narrativas de laboratório sobre programação e depuração;
- explicações operacionais de conceitos do DelphiLM;
- perguntas e respostas didáticas;
- mensagens fictícias de testes e diagnósticos.

O corpus não copia o manuscrito do livro, documentação da Embarcadero, obras literárias, páginas da internet ou dados de terceiros.

## Arquivos e uso

- `corpus-mini.txt`: deve permanecer entre 5 e 12 KiB; indicado para testes rápidos.
- `corpus-base.txt`: deve permanecer entre 350 e 600 KiB; indicado para treinamento e benchmark.

Os arquivos são regenerados por `tools/generate_corpus.py` e auditados por `tools/audit_corpus.py`. Os hashes SHA-256 ficam em `SHA256SUMS`.

## Limitações conhecidas

- O texto é tematicamente concentrado em programação, Delphi e modelos de linguagem.
- A construção por combinações determinísticas repete estruturas sintáticas mais do que um corpus natural.
- Nomes, situações e mensagens são fictícios.
- A distribuição de palavras não representa o português brasileiro em geral.
- Resultados favoráveis neste corpus não demonstram generalização para outros domínios.
- O modelo pode memorizar e reproduzir fragmentos do corpus.

## Conteúdo excluído

Não há dados pessoais, segredos, código de terceiros, textos obtidos por raspagem, material de leitores, conteúdo sensível intencional ou afirmações apresentadas como fatos sobre pessoas reais.

## Atualização

Mudanças de fragmentos, regras, tamanho ou licença exigem nova versão, novos hashes, nova auditoria e repetição dos benchmarks publicados.
