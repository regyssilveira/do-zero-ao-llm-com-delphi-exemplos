# DelphiLM Corpus Didático PT-BR

Corpus original criado para os experimentos do livro **Do Zero ao LLM com Delphi**.

## Arquivos

- `corpus-mini.txt`: execução rápida, testes e primeiros experimentos;
- `corpus-base.txt`: treinamento e benchmark editorial;
- `DATASET_CARD.md`: finalidade, composição, proveniência e limitações;
- `SHA256SUMS`: hashes que identificam a versão exata dos textos;
- `LICENSE`: Apache License 2.0 aplicada ao corpus.

## Regeneração

Na raiz do repositório:

```powershell
python tools/generate_corpus.py
python tools/audit_corpus.py
```

O gerador é determinístico. Qualquer alteração em conteúdo ou ordem modifica os hashes e deve produzir uma nova versão documentada do dataset.
