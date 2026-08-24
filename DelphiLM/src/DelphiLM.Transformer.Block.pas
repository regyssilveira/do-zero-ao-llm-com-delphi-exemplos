{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.Block;

interface

uses
  DelphiLM.Core.Random,
  DelphiLM.Math.Tensor,
  DelphiLM.Transformer.MultiHeadAttention;

type
  TLayerNorm = class
  private
    FBeta: TTensor;
    FDimension: Integer;
    FEpsilon: Single;
    FGamma: TTensor;
  public
    constructor Create(const ADimension: Integer;
      const AEpsilon: Single = 1.0E-5);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    procedure SetBeta(const AComponent: Integer; const AValue: Single);
    procedure SetGamma(const AComponent: Integer; const AValue: Single);
  end;

  TTransformerBlock = class
  private
    FAttention: TMultiHeadCausalSelfAttention;
    FEmbeddingDimension: Integer;
    FFeedForwardDimension: Integer;
    FFeedForwardInputBias: TTensor;
    FFeedForwardInputWeights: TTensor;
    FFeedForwardOutputBias: TTensor;
    FFeedForwardOutputWeights: TTensor;
    FLayerNorm1: TLayerNorm;
    FLayerNorm2: TLayerNorm;
    procedure InitializeWeights(const AWeights: TTensor;
      const AFanIn, AFanOut: Integer; var ARandom: TXorShift64Star);
  public
    constructor Create(const AEmbeddingDimension, AHeadCount,
      AFeedForwardDimension: Integer; var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    procedure SetFeedForwardOutputWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    property Attention: TMultiHeadCausalSelfAttention read FAttention;
  end;

implementation

uses
  DelphiLM.Neural.Layers,
  System.Math,
  System.SysUtils;

constructor TLayerNorm.Create(const ADimension: Integer;
  const AEpsilon: Single);
begin
  inherited Create;
  if ADimension <= 0 then
    raise EDelphiLMTensorError.Create('A dimensão da LayerNorm deve ser positiva.');
  if AEpsilon <= 0 then
    raise EDelphiLMTensorError.Create('Epsilon deve ser positivo.');
  FDimension := ADimension;
  FEpsilon := AEpsilon;
  FGamma := TTensor.Create([FDimension]);
  FBeta := TTensor.Create([FDimension]);
  FGamma.Fill(1);
end;

destructor TLayerNorm.Destroy;
begin
  FBeta.Free;
  FGamma.Free;
  inherited;
end;

function TLayerNorm.Forward(const AInput: TTensor): TTensor;
var
  Centered: Double;
  Component: Integer;
  InverseStandardDeviation: Double;
  Mean: Double;
  Position: Integer;
  Variance: Double;
begin
  if AInput = nil then
    raise EDelphiLMTensorError.Create('AInput não pode ser nil.');
  if (AInput.Rank <> 2) or (AInput.Dimension(1) <> FDimension) then
    raise EDelphiLMTensorError.CreateFmt(
      'LayerNorm esperava shape [T,%d].', [FDimension]);

  Result := TTensor.Create([AInput.Dimension(0), FDimension]);
  for Position := 0 to AInput.Dimension(0) - 1 do
  begin
    Mean := 0;
    for Component := 0 to FDimension - 1 do
      Mean := Mean + AInput.ValueAt([Position, Component]);
    Mean := Mean / FDimension;

    Variance := 0;
    for Component := 0 to FDimension - 1 do
    begin
      Centered := AInput.ValueAt([Position, Component]) - Mean;
      Variance := Variance + Centered * Centered;
    end;
    Variance := Variance / FDimension;
    InverseStandardDeviation := 1 / Sqrt(Variance + FEpsilon);

    for Component := 0 to FDimension - 1 do
      Result.SetValue(
        (AInput.ValueAt([Position, Component]) - Mean) *
        InverseStandardDeviation * FGamma.FlatValue(Component) +
        FBeta.FlatValue(Component), [Position, Component]);
  end;
end;

procedure TLayerNorm.SetBeta(const AComponent: Integer;
  const AValue: Single);
begin
  FBeta.SetFlatValue(AComponent, AValue);
end;

procedure TLayerNorm.SetGamma(const AComponent: Integer;
  const AValue: Single);
begin
  FGamma.SetFlatValue(AComponent, AValue);
end;

constructor TTransformerBlock.Create(const AEmbeddingDimension,
  AHeadCount, AFeedForwardDimension: Integer;
  var ARandom: TXorShift64Star);
begin
  inherited Create;
  if AFeedForwardDimension <= 0 then
    raise EDelphiLMTensorError.Create(
      'A dimensão do feed-forward deve ser positiva.');
  FEmbeddingDimension := AEmbeddingDimension;
  FFeedForwardDimension := AFeedForwardDimension;
  FLayerNorm1 := TLayerNorm.Create(FEmbeddingDimension);
  FLayerNorm2 := TLayerNorm.Create(FEmbeddingDimension);
  FAttention := TMultiHeadCausalSelfAttention.Create(
    FEmbeddingDimension, AHeadCount, ARandom);
  FFeedForwardInputWeights := TTensor.Create([
    FFeedForwardDimension, FEmbeddingDimension]);
  FFeedForwardInputBias := TTensor.Create([FFeedForwardDimension]);
  FFeedForwardOutputWeights := TTensor.Create([
    FEmbeddingDimension, FFeedForwardDimension]);
  FFeedForwardOutputBias := TTensor.Create([FEmbeddingDimension]);
  InitializeWeights(FFeedForwardInputWeights, FEmbeddingDimension,
    FFeedForwardDimension, ARandom);
  InitializeWeights(FFeedForwardOutputWeights, FFeedForwardDimension,
    FEmbeddingDimension, ARandom);
end;

destructor TTransformerBlock.Destroy;
begin
  FFeedForwardOutputBias.Free;
  FFeedForwardOutputWeights.Free;
  FFeedForwardInputBias.Free;
  FFeedForwardInputWeights.Free;
  FAttention.Free;
  FLayerNorm2.Free;
  FLayerNorm1.Free;
  inherited;
end;

procedure TTransformerBlock.InitializeWeights(const AWeights: TTensor;
  const AFanIn, AFanOut: Integer; var ARandom: TXorShift64Star);
var
  InputIndex: Integer;
  Limit: Single;
  OutputIndex: Integer;
begin
  Limit := Sqrt(6.0 / (AFanIn + AFanOut));
  for OutputIndex := 0 to AWeights.Dimension(0) - 1 do
    for InputIndex := 0 to AWeights.Dimension(1) - 1 do
      AWeights.SetValue((2 * ARandom.NextSingle - 1) * Limit,
        [OutputIndex, InputIndex]);
end;

function TTransformerBlock.Forward(const AInput: TTensor): TTensor;
var
  AttentionOutput: TTensor;
  Component: Integer;
  FeedForwardOutput: TTensor;
  Hidden: TTensor;
  HiddenComponent: Integer;
  Normalized: TTensor;
  Position: Integer;
  Residual: TTensor;
  Sum: Single;
begin
  if AInput = nil then
    raise EDelphiLMTensorError.Create('AInput não pode ser nil.');
  if (AInput.Rank <> 2) or
     (AInput.Dimension(1) <> FEmbeddingDimension) then
    raise EDelphiLMTensorError.CreateFmt(
      'Bloco Transformer esperava shape [T,%d].', [FEmbeddingDimension]);

  Normalized := FLayerNorm1.Forward(AInput);
  AttentionOutput := nil;
  Residual := nil;
  Hidden := nil;
  FeedForwardOutput := nil;
  try
    AttentionOutput := FAttention.Forward(Normalized);
    Residual := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
    for Position := 0 to AInput.Dimension(0) - 1 do
      for Component := 0 to FEmbeddingDimension - 1 do
        Residual.SetValue(AInput.ValueAt([Position, Component]) +
          AttentionOutput.ValueAt([Position, Component]),
          [Position, Component]);

    FreeAndNil(Normalized);
    Normalized := FLayerNorm2.Forward(Residual);
    Hidden := TTensor.Create([
      AInput.Dimension(0), FFeedForwardDimension]);
    for Position := 0 to AInput.Dimension(0) - 1 do
      for HiddenComponent := 0 to FFeedForwardDimension - 1 do
      begin
        Sum := FFeedForwardInputBias.FlatValue(HiddenComponent);
        for Component := 0 to FEmbeddingDimension - 1 do
          Sum := Sum + FFeedForwardInputWeights.ValueAt([
            HiddenComponent, Component]) *
            Normalized.ValueAt([Position, Component]);
        Hidden.SetValue(GELU(Sum), [Position, HiddenComponent]);
      end;

    FeedForwardOutput := TTensor.Create([
      AInput.Dimension(0), FEmbeddingDimension]);
    for Position := 0 to AInput.Dimension(0) - 1 do
      for Component := 0 to FEmbeddingDimension - 1 do
      begin
        Sum := FFeedForwardOutputBias.FlatValue(Component);
        for HiddenComponent := 0 to FFeedForwardDimension - 1 do
          Sum := Sum + FFeedForwardOutputWeights.ValueAt([
            Component, HiddenComponent]) *
            Hidden.ValueAt([Position, HiddenComponent]);
        FeedForwardOutput.SetValue(Sum, [Position, Component]);
      end;

    Result := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
    for Position := 0 to AInput.Dimension(0) - 1 do
      for Component := 0 to FEmbeddingDimension - 1 do
        Result.SetValue(Residual.ValueAt([Position, Component]) +
          FeedForwardOutput.ValueAt([Position, Component]),
          [Position, Component]);
  finally
    FeedForwardOutput.Free;
    Hidden.Free;
    Residual.Free;
    AttentionOutput.Free;
    Normalized.Free;
  end;
end;

procedure TTransformerBlock.SetFeedForwardOutputWeight(
  const AOutput, AInput: Integer; const AValue: Single);
begin
  FFeedForwardOutputWeights.SetValue(AValue, [AOutput, AInput]);
end;

end.
