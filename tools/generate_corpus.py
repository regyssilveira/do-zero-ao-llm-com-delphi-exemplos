from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

PEOPLE = [
    "Ana", "Bruno", "Clara", "Daniel", "Elisa", "Fábio", "Helena", "Igor",
    "Joana", "Lucas", "Marina", "Nuno", "Olívia", "Paulo", "Rita", "Sérgio",
]

MOMENTS = [
    "no início da manhã", "antes da primeira compilação", "durante a revisão",
    "depois de uma pequena pausa", "ao repetir o experimento", "no fim da tarde",
    "antes de salvar o checkpoint", "durante a execução em Release",
]

ACTIONS = [
    "abriu o projeto e conferiu a configuração",
    "executou a suíte e leu cada mensagem",
    "comparou a conta feita no papel com a saída",
    "reduziu o exemplo até restar uma única operação",
    "inspecionou o shape antes de percorrer o tensor",
    "repetiu o treinamento com a mesma seed",
    "verificou quem era responsável por liberar cada objeto",
    "registrou a loss antes e depois da atualização",
]

PROBLEMS = [
    "um índice apontava para a posição seguinte",
    "o contexto ultrapassava o limite configurado",
    "um gradiente antigo não havia sido zerado",
    "a entrada continha uma forma Unicode diferente",
    "a taxa de aprendizado era grande para aquele teste",
    "o arquivo de parâmetros usava outra configuração",
    "a máscara permitia consultar uma posição futura",
    "a soma acumulava com precisão insuficiente",
]

SOLUTIONS = [
    "transformou a expectativa em uma asserção pequena",
    "imprimiu as dimensões e acompanhou os índices",
    "fixou a seed e eliminou uma fonte de variação",
    "normalizou o texto antes de criar o vocabulário",
    "comparou Debug e Release com a mesma entrada",
    "rejeitou o estado inválido com uma mensagem clara",
    "guardou somente os valores necessários ao backward",
    "acumulou a operação sensível usando Double",
]

RESULTS = [
    "O teste passou e o motivo ficou visível.",
    "A saída voltou a coincidir com a conta manual.",
    "A falha deixou de aparecer nas execuções seguintes.",
    "O programa terminou com código zero e resultado reproduzível.",
    "A loss diminuiu sem esconder a origem da correção.",
    "O exemplo ficou menor, mais claro e mais fácil de ensinar.",
    "A mensagem permitiu localizar o erro sem adivinhação.",
    "O comportamento permaneceu igual nos dois perfis de compilação.",
]

CONCEPTS = [
    ("token", "uma unidade inteira que identifica parte do texto", "não representa tamanho nem importância"),
    ("vocabulário", "o mapeamento entre unidades textuais e identificadores", "precisa permanecer igual ao carregar um checkpoint"),
    ("embedding", "um vetor aprendido selecionado por um token ou posição", "ganha utilidade quando o treinamento ajusta seus componentes"),
    ("shape", "a lista que informa o tamanho de cada eixo", "funciona como legenda para interpretar os índices"),
    ("stride", "o salto linear associado a um eixo do tensor", "permite localizar coordenadas dentro de um buffer contíguo"),
    ("logit", "uma pontuação ainda não normalizada", "pode ser negativo e não precisa somar um"),
    ("Softmax", "a operação que transforma logits em valores positivos que somam um", "deve ser calculada de modo numericamente estável"),
    ("loss", "um número que resume o erro segundo o objetivo de treino", "não mede inteligência nem correção factual"),
    ("gradiente", "a informação local sobre como a loss reage a uma mudança", "orienta a atualização dos parâmetros"),
    ("backward", "o percurso que leva gradientes da saída para a entrada", "reutiliza valores guardados durante o forward"),
    ("Adam", "um otimizador que mantém médias do histórico dos gradientes", "adapta o passo aplicado a cada parâmetro"),
    ("máscara causal", "a restrição que bloqueia posições futuras", "impede que o treinamento consulte a resposta"),
    ("atenção", "uma combinação ponderada de valores do contexto", "usa consultas e chaves para calcular compatibilidades"),
    ("LayerNorm", "a normalização aplicada aos componentes de cada posição", "possui escala e deslocamento aprendidos"),
    ("conexão residual", "o caminho que soma a entrada à transformação", "ajuda a preservar informação entre os blocos"),
    ("checkpoint", "um arquivo versionado com parâmetros e metadados", "deve rejeitar modelos incompatíveis"),
]

QUESTIONS = [
    ("Por que usar Win64?", "Porque o treinamento mantém pesos, gradientes, ativações e estados do otimizador, e o espaço de endereçamento adicional reduz limitações artificiais."),
    ("Por que fixar uma seed?", "Porque a mesma sequência pseudoaleatória facilita comparar duas execuções e localizar mudanças reais no código."),
    ("Por que começar com caracteres?", "Porque o vocabulário cabe na tela, o round-trip é observável e o algoritmo não esconde unidades de subpalavra."),
    ("Por que testar números pequenos?", "Porque uma conta que cabe no papel permite separar erro matemático, índice incorreto e configuração inválida."),
    ("Por que devolver logits?", "Porque treinamento e geração aplicam transformações diferentes antes de obter probabilidades."),
    ("Por que limitar gradientes?", "Porque um gradiente excepcionalmente grande pode desestabilizar várias atualizações seguintes."),
    ("Por que salvar metadados?", "Porque os mesmos bytes de pesos não descrevem sozinhos vocabulário, dimensões e versão do formato."),
    ("Por que medir em Release?", "Porque otimizações e verificações do compilador podem alterar tempo e revelar dependências indevidas do perfil."),
]

MESSAGES = [
    "[OK] configuração validada antes da alocação",
    "[OK] texto normalizado recuperado após encode e decode",
    "[OK] tensor percorreu o buffer em ordem de linha",
    "[OK] gradiente analítico coincidiu com diferenças finitas",
    "[OK] posição futura não alterou saídas anteriores",
    "[OK] checkpoint restaurou logits idênticos",
    "[OK] mesma seed produziu a mesma sequência",
    "[OK] loss final ficou abaixo da loss inicial",
]


def narrative(index: int) -> str:
    person = PEOPLE[index % len(PEOPLE)]
    moment = MOMENTS[(index // 2) % len(MOMENTS)]
    action = ACTIONS[(index // 3) % len(ACTIONS)]
    problem = PROBLEMS[(index // 5) % len(PROBLEMS)]
    solution = SOLUTIONS[(index // 7) % len(SOLUTIONS)]
    result = RESULTS[(index // 11) % len(RESULTS)]
    return (
        f"{moment.capitalize()}, {person} {action}. Durante a conferência, percebeu que "
        f"{problem}. Em vez de alterar várias partes ao mesmo tempo, {solution}. {result} "
        "A equipe registrou a causa, a entrada usada e o resultado esperado para que outra "
        "pessoa pudesse repetir a investigação."
    )


def concept(index: int) -> str:
    name, definition, consequence = CONCEPTS[index % len(CONCEPTS)]
    return (
        f"No laboratório, {name} significa {definition}. Essa definição é operacional: "
        f"{consequence}. O leitor pode observar o conceito em uma entrada pequena, conferir "
        "a transformação e somente depois generalizar o procedimento para o modelo completo."
    )


def question(index: int) -> str:
    title, answer = QUESTIONS[index % len(QUESTIONS)]
    return (
        f"Pergunta do laboratório: {title}\nResposta: {answer} A decisão permanece explícita "
        "no código, nos testes e no registro do experimento."
    )


def diagnostic(index: int) -> str:
    first = MESSAGES[index % len(MESSAGES)]
    second = MESSAGES[(index + 3) % len(MESSAGES)]
    return (
        f"Registro de execução:\n{first}\n{second}\n"
        "Duas linhas aprovadas não provam todo o sistema, mas documentam propriedades concretas. "
        "Se uma delas falhar, a mensagem deve indicar qual contrato deixou de ser atendido."
    )


def build_paragraphs(count: int) -> list[str]:
    builders = (narrative, narrative, concept, narrative, question, concept, narrative, diagnostic)
    return [builders[index % len(builders)](index) for index in range(count)]


def normalized_text(paragraphs: list[str]) -> str:
    heading = (
        "DelphiLM Corpus Didático PT-BR\n"
        "Textos originais sobre programação, testes e modelos de linguagem.\n"
    )
    return unicodedata.normalize("NFC", heading + "\n" + "\n\n".join(paragraphs) + "\n")


def write_dataset(name: str, paragraphs: list[str]) -> Path:
    path = DATASETS / name
    path.write_text(normalized_text(paragraphs), encoding="utf-8", newline="\n")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    mini = write_dataset("corpus-mini.txt", build_paragraphs(28))
    base = write_dataset("corpus-base.txt", build_paragraphs(1500))

    mini_size = mini.stat().st_size
    base_size = base.stat().st_size
    if not 5 * 1024 <= mini_size <= 12 * 1024:
        raise RuntimeError(f"corpus-mini fora do intervalo: {mini_size} bytes")
    if not 350 * 1024 <= base_size <= 600 * 1024:
        raise RuntimeError(f"corpus-base fora do intervalo: {base_size} bytes")

    checksums = (
        f"{sha256(base)}  {base.name}\n"
        f"{sha256(mini)}  {mini.name}\n"
    )
    (DATASETS / "SHA256SUMS").write_text(checksums, encoding="ascii", newline="\n")
    print(f"{mini.name}: {mini_size} bytes")
    print(f"{base.name}: {base_size} bytes")


if __name__ == "__main__":
    main()
