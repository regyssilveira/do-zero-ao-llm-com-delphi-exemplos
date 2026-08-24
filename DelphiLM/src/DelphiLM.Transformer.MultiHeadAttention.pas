{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.MultiHeadAttention;

interface

uses
  DelphiLM.Core.Random,
  DelphiLM.Math.Tensor;

type
  TMultiHeadCausalSelfAttention = class
  private
    FEmbeddingDimension: Integer;
    FHeadCount: Integer;
    FHeadDimension: Integer;
    FKeyBias: TTensor;
    FKeyWeights: TTensor;
    FOutputBias: TTensor;
    FOutputWeights: TTensor;
    FQueryBias: TTensor;
    FQueryWeights: TTensor;
    FValueBias: TTensor;
    FValueWeights: TTensor;
    procedure InitializeWeights(const AWeights: TTensor;
      var ARandom: TXorShift64Star);
    function ProjectHead(const AInput, AWeights, ABias: TTensor;
      const AHead: Integer): TTensor;
  public
    constructor Create(const AEmbeddingDimension, AHeadCount: Integer;
      var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    procedure SetQueryWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    procedure SetKeyWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    procedure SetValueWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    procedure SetOutputWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    property EmbeddingDimension: Integer read FEmbeddingDimension;
    property HeadCount: Integer read FHeadCount;
    property HeadDimension: Integer read FHeadDimension;
  end;

implementation

uses
  DelphiLM.Transformer.Attention,
  System.Math;

constructor TMultiHeadCausalSelfAttention.Create(
  const AEmbeddingDimension, AHeadCount: Integer;
  var ARandom: TXorShift64Star);
begin
  inherited Create;
  if AEmbeddingDimension <= 0 then
    raise EDelphiLMTensorError.Create(
      'EmbeddingDimension deve ser positivo.');
  if AHeadCount <= 0 then
    raise EDelphiLMTensorError.Create('HeadCount deve ser positivo.');
  if (AEmbeddingDimension mod AHeadCount) <> 0 then
    raise EDelphiLMTensorError.Create(
      'EmbeddingDimension deve ser divisível por HeadCount.');

  FEmbeddingDimension := AEmbeddingDimension;
  FHeadCount := AHeadCount;
  FHeadDimension := AEmbeddingDimension div AHeadCount;
  FQueryWeights := TTensor.Create([
    FEmbeddingDimension, FEmbeddingDimension]);
  FKeyWeights := TTensor.Create([
    FEmbeddingDimension, FEmbeddingDimension]);
  FValueWeights := TTensor.Create([
    FEmbeddingDimension, FEmbeddingDimension]);
  FOutputWeights := TTensor.Create([
    FEmbeddingDimension, FEmbeddingDimension]);
  FQueryBias := TTensor.Create([FEmbeddingDimension]);
  FKeyBias := TTensor.Create([FEmbeddingDimension]);
  FValueBias := TTensor.Create([FEmbeddingDimension]);
  FOutputBias := TTensor.Create([FEmbeddingDimension]);
  InitializeWeights(FQueryWeights, ARandom);
  InitializeWeights(FKeyWeights, ARandom);
  InitializeWeights(FValueWeights, ARandom);
  InitializeWeights(FOutputWeights, ARandom);
end;

destructor TMultiHeadCausalSelfAttention.Destroy;
begin
  FOutputBias.Free;
  FValueBias.Free;
  FKeyBias.Free;
  FQueryBias.Free;
  FOutputWeights.Free;
  FValueWeights.Free;
  FKeyWeights.Free;
  FQueryWeights.Free;
  inherited;
end;

procedure TMultiHeadCausalSelfAttention.InitializeWeights(
  const AWeights: TTensor; var ARandom: TXorShift64Star);
var
  InputIndex: Integer;
  Limit: Single;
  OutputIndex: Integer;
begin
  Limit := Sqrt(6.0 / (2 * FEmbeddingDimension));
  for OutputIndex := 0 to FEmbeddingDimension - 1 do
    for InputIndex := 0 to FEmbeddingDimension - 1 do
      AWeights.SetValue((2 * ARandom.NextSingle - 1) * Limit,
        [OutputIndex, InputIndex]);
end;

function TMultiHeadCausalSelfAttention.ProjectHead(const AInput,
  AWeights, ABias: TTensor; const AHead: Integer): TTensor;
var
  Component: Integer;
  InputComponent: Integer;
  OutputComponent: Integer;
  Position: Integer;
  Sum: Single;
begin
  Result := TTensor.Create([AInput.Dimension(0), FHeadDimension]);
  for Position := 0 to AInput.Dimension(0) - 1 do
    for Component := 0 to FHeadDimension - 1 do
    begin
      OutputComponent := AHead * FHeadDimension + Component;
      Sum := ABias.FlatValue(OutputComponent);
      for InputComponent := 0 to FEmbeddingDimension - 1 do
        Sum := Sum + AWeights.ValueAt([
          OutputComponent, InputComponent]) *
          AInput.ValueAt([Position, InputComponent]);
      Result.SetValue(Sum, [Position, Component]);
    end;
end;

function TMultiHeadCausalSelfAttention.Forward(
  const AInput: TTensor): TTensor;
var
  AttentionOutput: TTensor;
  AttentionWeights: TTensor;
  Component: Integer;
  Concatenated: TTensor;
  Head: Integer;
  InputComponent: Integer;
  Keys: TTensor;
  Position: Integer;
  Queries: TTensor;
  Sum: Single;
  Values: TTensor;
begin
  if AInput = nil then
    raise EDelphiLMTensorError.Create('AInput não pode ser nil.');
  if (AInput.Rank <> 2) or
     (AInput.Dimension(1) <> FEmbeddingDimension) then
    raise EDelphiLMTensorError.CreateFmt(
      'Self-attention esperava shape [T,%d].', [FEmbeddingDimension]);

  Concatenated := TTensor.Create([
    AInput.Dimension(0), FEmbeddingDimension]);
  try
    for Head := 0 to FHeadCount - 1 do
    begin
      Queries := ProjectHead(AInput, FQueryWeights, FQueryBias, Head);
      Keys := ProjectHead(AInput, FKeyWeights, FKeyBias, Head);
      Values := ProjectHead(AInput, FValueWeights, FValueBias, Head);
      AttentionOutput := nil;
      AttentionWeights := nil;
      try
        AttentionOutput := CausalScaledDotProductAttention(
          Queries, Keys, Values, AttentionWeights);
        for Position := 0 to AInput.Dimension(0) - 1 do
          for Component := 0 to FHeadDimension - 1 do
            Concatenated.SetValue(
              AttentionOutput.ValueAt([Position, Component]),
              [Position, Head * FHeadDimension + Component]);
      finally
        AttentionWeights.Free;
        AttentionOutput.Free;
        Values.Free;
        Keys.Free;
        Queries.Free;
      end;
    end;

    Result := TTensor.Create([
      AInput.Dimension(0), FEmbeddingDimension]);
    for Position := 0 to AInput.Dimension(0) - 1 do
      for Component := 0 to FEmbeddingDimension - 1 do
      begin
        Sum := FOutputBias.FlatValue(Component);
        for InputComponent := 0 to FEmbeddingDimension - 1 do
          Sum := Sum + FOutputWeights.ValueAt([
            Component, InputComponent]) *
            Concatenated.ValueAt([Position, InputComponent]);
        Result.SetValue(Sum, [Position, Component]);
      end;
  finally
    Concatenated.Free;
  end;
end;

procedure TMultiHeadCausalSelfAttention.SetQueryWeight(
  const AOutput, AInput: Integer; const AValue: Single);
begin
  FQueryWeights.SetValue(AValue, [AOutput, AInput]);
end;

procedure TMultiHeadCausalSelfAttention.SetKeyWeight(
  const AOutput, AInput: Integer; const AValue: Single);
begin
  FKeyWeights.SetValue(AValue, [AOutput, AInput]);
end;

procedure TMultiHeadCausalSelfAttention.SetValueWeight(
  const AOutput, AInput: Integer; const AValue: Single);
begin
  FValueWeights.SetValue(AValue, [AOutput, AInput]);
end;

procedure TMultiHeadCausalSelfAttention.SetOutputWeight(
  const AOutput, AInput: Integer; const AValue: Single);
begin
  FOutputWeights.SetValue(AValue, [AOutput, AInput]);
end;

end.
