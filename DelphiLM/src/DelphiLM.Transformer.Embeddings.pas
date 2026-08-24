{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.Embeddings;

interface

uses
  DelphiLM.Core.Random,
  DelphiLM.Math.Tensor;

type
  TEmbeddingTable = class
  private
    FEmbeddingDimension: Integer;
    FGradients: TTensor;
    FItemCount: Integer;
    FWeights: TTensor;
    procedure ValidateItem(const AItem: Integer);
  public
    constructor Create(const AItemCount, AEmbeddingDimension: Integer;
      var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Lookup(const AItems: TArray<Integer>): TTensor;
    procedure AccumulateGradients(const AItems: TArray<Integer>;
      const AGradOutput: TTensor);
    procedure ZeroGrad;
    procedure SetValue(const AItem, AComponent: Integer;
      const AValue: Single);
    function ValueAt(const AItem, AComponent: Integer): Single;
    function GradientAt(const AItem, AComponent: Integer): Single;
    property ItemCount: Integer read FItemCount;
    property EmbeddingDimension: Integer read FEmbeddingDimension;
  end;

  TTokenPositionEmbedding = class
  private
    FCachedTokens: TArray<Integer>;
    FMaxContextLength: Integer;
    FPositionTable: TEmbeddingTable;
    FTokenTable: TEmbeddingTable;
  public
    constructor Create(const AVocabularySize, AMaxContextLength,
      AEmbeddingDimension: Integer; var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const ATokens: TArray<Integer>): TTensor;
    procedure Backward(const AGradOutput: TTensor);
    procedure ZeroGrad;
    property TokenTable: TEmbeddingTable read FTokenTable;
    property PositionTable: TEmbeddingTable read FPositionTable;
  end;

implementation

uses
  System.Math,
  System.SysUtils;

constructor TEmbeddingTable.Create(const AItemCount,
  AEmbeddingDimension: Integer; var ARandom: TXorShift64Star);
var
  Component: Integer;
  Item: Integer;
  Limit: Single;
begin
  inherited Create;
  if AItemCount <= 0 then
    raise EDelphiLMTensorError.Create('ItemCount deve ser positivo.');
  if AEmbeddingDimension <= 0 then
    raise EDelphiLMTensorError.Create(
      'EmbeddingDimension deve ser positivo.');
  FItemCount := AItemCount;
  FEmbeddingDimension := AEmbeddingDimension;
  FWeights := TTensor.Create([FItemCount, FEmbeddingDimension]);
  FGradients := TTensor.Create([FItemCount, FEmbeddingDimension]);
  Limit := 1 / Sqrt(FEmbeddingDimension);
  for Item := 0 to FItemCount - 1 do
    for Component := 0 to FEmbeddingDimension - 1 do
      FWeights.SetValue((2 * ARandom.NextSingle - 1) * Limit,
        [Item, Component]);
end;

destructor TEmbeddingTable.Destroy;
begin
  FGradients.Free;
  FWeights.Free;
  inherited;
end;

procedure TEmbeddingTable.ValidateItem(const AItem: Integer);
begin
  if (AItem < 0) or (AItem >= FItemCount) then
    raise EDelphiLMTensorError.CreateFmt(
      'Item %d fora da tabela de tamanho %d.', [AItem, FItemCount]);
end;

function TEmbeddingTable.Lookup(const AItems: TArray<Integer>): TTensor;
var
  Component: Integer;
  ItemIndex: Integer;
begin
  if Length(AItems) = 0 then
    raise EDelphiLMTensorError.Create('Lookup exige ao menos um item.');
  Result := TTensor.Create([Length(AItems), FEmbeddingDimension]);
  for ItemIndex := 0 to High(AItems) do
  begin
    ValidateItem(AItems[ItemIndex]);
    for Component := 0 to FEmbeddingDimension - 1 do
      Result.SetValue(FWeights.ValueAt([AItems[ItemIndex], Component]),
        [ItemIndex, Component]);
  end;
end;

procedure TEmbeddingTable.AccumulateGradients(
  const AItems: TArray<Integer>; const AGradOutput: TTensor);
var
  Component: Integer;
  ItemIndex: Integer;
begin
  if AGradOutput = nil then
    raise EArgumentNilException.Create('AGradOutput');
  if (AGradOutput.Rank <> 2) or
     (AGradOutput.Dimension(0) <> Length(AItems)) or
     (AGradOutput.Dimension(1) <> FEmbeddingDimension) then
    raise EDelphiLMTensorError.Create(
      'Gradiente incompatível com o lookup de embeddings.');
  for ItemIndex := 0 to High(AItems) do
  begin
    ValidateItem(AItems[ItemIndex]);
    for Component := 0 to FEmbeddingDimension - 1 do
      FGradients.SetValue(
        FGradients.ValueAt([AItems[ItemIndex], Component]) +
          AGradOutput.ValueAt([ItemIndex, Component]),
        [AItems[ItemIndex], Component]);
  end;
end;

procedure TEmbeddingTable.ZeroGrad;
begin
  FGradients.Fill(0);
end;

procedure TEmbeddingTable.SetValue(const AItem, AComponent: Integer;
  const AValue: Single);
begin
  ValidateItem(AItem);
  FWeights.SetValue(AValue, [AItem, AComponent]);
end;

function TEmbeddingTable.ValueAt(const AItem,
  AComponent: Integer): Single;
begin
  ValidateItem(AItem);
  Result := FWeights.ValueAt([AItem, AComponent]);
end;

function TEmbeddingTable.GradientAt(const AItem,
  AComponent: Integer): Single;
begin
  ValidateItem(AItem);
  Result := FGradients.ValueAt([AItem, AComponent]);
end;

constructor TTokenPositionEmbedding.Create(const AVocabularySize,
  AMaxContextLength, AEmbeddingDimension: Integer;
  var ARandom: TXorShift64Star);
begin
  inherited Create;
  if AMaxContextLength <= 0 then
    raise EDelphiLMTensorError.Create(
      'MaxContextLength deve ser positivo.');
  FMaxContextLength := AMaxContextLength;
  FTokenTable := TEmbeddingTable.Create(AVocabularySize,
    AEmbeddingDimension, ARandom);
  FPositionTable := TEmbeddingTable.Create(AMaxContextLength,
    AEmbeddingDimension, ARandom);
end;

destructor TTokenPositionEmbedding.Destroy;
begin
  FPositionTable.Free;
  FTokenTable.Free;
  inherited;
end;

function TTokenPositionEmbedding.Forward(
  const ATokens: TArray<Integer>): TTensor;
var
  Component: Integer;
  Position: Integer;
begin
  if Length(ATokens) = 0 then
    raise EDelphiLMTensorError.Create('A sequência não pode estar vazia.');
  if Length(ATokens) > FMaxContextLength then
    raise EDelphiLMTensorError.CreateFmt(
      'Sequência de %d tokens excede o contexto máximo %d.',
      [Length(ATokens), FMaxContextLength]);
  FCachedTokens := Copy(ATokens);
  Result := TTensor.Create([
    Length(ATokens), FTokenTable.EmbeddingDimension]);
  for Position := 0 to High(ATokens) do
  begin
    FTokenTable.ValidateItem(ATokens[Position]);
    for Component := 0 to FTokenTable.EmbeddingDimension - 1 do
      Result.SetValue(
        FTokenTable.ValueAt(ATokens[Position], Component) +
          FPositionTable.ValueAt(Position, Component),
        [Position, Component]);
  end;
end;

procedure TTokenPositionEmbedding.Backward(const AGradOutput: TTensor);
var
  Positions: TArray<Integer>;
  Position: Integer;
begin
  if Length(FCachedTokens) = 0 then
    raise EDelphiLMTensorError.Create('Backward exige um Forward anterior.');
  SetLength(Positions, Length(FCachedTokens));
  for Position := 0 to High(Positions) do
    Positions[Position] := Position;
  FTokenTable.AccumulateGradients(FCachedTokens, AGradOutput);
  FPositionTable.AccumulateGradients(Positions, AGradOutput);
end;

procedure TTokenPositionEmbedding.ZeroGrad;
begin
  FTokenTable.ZeroGrad;
  FPositionTable.ZeroGrad;
end;

end.
