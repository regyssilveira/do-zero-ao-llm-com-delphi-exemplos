{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.Block;

interface

uses
  DelphiLM.Core.Random, DelphiLM.Math.Tensor, DelphiLM.Neural.Parameters,
  DelphiLM.Transformer.MultiHeadAttention, System.Generics.Collections;

type
  TLayerNorm = class
  private
    FBeta, FGamma: TTrainableParameter;
    FCachedInverseStd, FCachedNormalized: TTensor;
    FDimension: Integer;
    FEpsilon: Single;
    procedure ClearCache;
  public
    constructor Create(const ADimension: Integer; const AEpsilon: Single = 1.0E-5);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    function Backward(const AGradOutput: TTensor): TTensor;
    procedure CollectParameters(const AList: TList<TTrainableParameter>);
    procedure ZeroGrad;
    procedure SetBeta(const AComponent: Integer; const AValue: Single);
    procedure SetGamma(const AComponent: Integer; const AValue: Single);
  end;

  TTransformerBlock = class
  private
    FAttention: TMultiHeadCausalSelfAttention;
    FCachedHiddenPre, FCachedNormalized2, FCachedResidual: TTensor;
    FEmbeddingDimension, FFeedForwardDimension: Integer;
    FFeedForwardInputBias, FFeedForwardInputWeights,
    FFeedForwardOutputBias, FFeedForwardOutputWeights: TTrainableParameter;
    FLayerNorm1, FLayerNorm2: TLayerNorm;
    procedure ClearCache;
    procedure InitializeWeights(const AWeights: TTrainableParameter;
      const AFanIn, AFanOut: Integer; var ARandom: TXorShift64Star);
  public
    constructor Create(const AEmbeddingDimension, AHeadCount,
      AFeedForwardDimension: Integer; var ARandom: TXorShift64Star);
    destructor Destroy; override;
    function Forward(const AInput: TTensor): TTensor;
    function Backward(const AGradOutput: TTensor): TTensor;
    procedure CollectParameters(const AList: TList<TTrainableParameter>);
    procedure ZeroGrad;
    procedure SetFeedForwardOutputWeight(const AOutput, AInput: Integer;
      const AValue: Single);
    property Attention: TMultiHeadCausalSelfAttention read FAttention;
  end;

implementation

uses
  DelphiLM.Neural.Layers, System.Math, System.SysUtils;

constructor TLayerNorm.Create(const ADimension: Integer; const AEpsilon: Single);
begin
  inherited Create;
  if ADimension <= 0 then raise EDelphiLMTensorError.Create(
    'A dimensão da LayerNorm deve ser positiva.');
  if AEpsilon <= 0 then raise EDelphiLMTensorError.Create('Epsilon deve ser positivo.');
  FDimension := ADimension; FEpsilon := AEpsilon;
  FGamma := TTrainableParameter.Create([FDimension]);
  FBeta := TTrainableParameter.Create([FDimension]);
  FGamma.Value.Fill(1);
end;

destructor TLayerNorm.Destroy;
begin ClearCache; FBeta.Free; FGamma.Free; inherited end;

procedure TLayerNorm.ClearCache;
begin FreeAndNil(FCachedInverseStd); FreeAndNil(FCachedNormalized) end;

function TLayerNorm.Forward(const AInput: TTensor): TTensor;
var C, P: Integer; Centered, InvStd, Mean, Variance: Double;
begin
  if (AInput = nil) or (AInput.Rank <> 2) or
     (AInput.Dimension(1) <> FDimension) then
    raise EDelphiLMTensorError.CreateFmt('LayerNorm esperava shape [T,%d].', [FDimension]);
  ClearCache;
  FCachedNormalized := TTensor.Create([AInput.Dimension(0), FDimension]);
  FCachedInverseStd := TTensor.Create([AInput.Dimension(0)]);
  Result := TTensor.Create([AInput.Dimension(0), FDimension]);
  for P := 0 to AInput.Dimension(0) - 1 do
  begin
    Mean := 0;
    for C := 0 to FDimension - 1 do Mean := Mean + AInput.ValueAt([P, C]);
    Mean := Mean / FDimension; Variance := 0;
    for C := 0 to FDimension - 1 do
    begin Centered := AInput.ValueAt([P, C]) - Mean; Variance := Variance + Centered * Centered end;
    InvStd := 1 / Sqrt(Variance / FDimension + FEpsilon);
    FCachedInverseStd.SetFlatValue(P, InvStd);
    for C := 0 to FDimension - 1 do
    begin
      Centered := (AInput.ValueAt([P, C]) - Mean) * InvStd;
      FCachedNormalized.SetValue(Centered, [P, C]);
      Result.SetValue(Centered * FGamma.Value.FlatValue(C) +
        FBeta.Value.FlatValue(C), [P, C]);
    end;
  end;
end;

function TLayerNorm.Backward(const AGradOutput: TTensor): TTensor;
var C, P: Integer; DY, SumDY, SumDYN: Double;
begin
  if FCachedNormalized = nil then raise EDelphiLMTensorError.Create(
    'Backward da LayerNorm exige um Forward anterior.');
  Result := TTensor.Create([FCachedNormalized.Dimension(0), FDimension]);
  for P := 0 to FCachedNormalized.Dimension(0) - 1 do
  begin
    SumDY := 0; SumDYN := 0;
    for C := 0 to FDimension - 1 do
    begin
      DY := AGradOutput.ValueAt([P, C]) * FGamma.Value.FlatValue(C);
      SumDY := SumDY + DY;
      SumDYN := SumDYN + DY * FCachedNormalized.ValueAt([P, C]);
      FBeta.AddGradientFlat(C, AGradOutput.ValueAt([P, C]));
      FGamma.AddGradientFlat(C, AGradOutput.ValueAt([P, C]) *
        FCachedNormalized.ValueAt([P, C]));
    end;
    for C := 0 to FDimension - 1 do
    begin
      DY := AGradOutput.ValueAt([P, C]) * FGamma.Value.FlatValue(C);
      Result.SetValue(FCachedInverseStd.FlatValue(P) / FDimension *
        (FDimension * DY - SumDY - FCachedNormalized.ValueAt([P, C]) * SumDYN), [P, C]);
    end;
  end;
end;

procedure TLayerNorm.CollectParameters(const AList: TList<TTrainableParameter>);
begin AList.Add(FGamma); AList.Add(FBeta) end;
procedure TLayerNorm.ZeroGrad; begin FGamma.ZeroGrad; FBeta.ZeroGrad end;
procedure TLayerNorm.SetBeta(const AComponent: Integer; const AValue: Single);
begin FBeta.Value.SetFlatValue(AComponent, AValue) end;
procedure TLayerNorm.SetGamma(const AComponent: Integer; const AValue: Single);
begin FGamma.Value.SetFlatValue(AComponent, AValue) end;

function GELUDerivative(const X: Single): Single;
const K: Double = 0.7978845608;
var T, U: Double;
begin
  U := K * (X + 0.044715 * X * X * X); T := Tanh(U);
  Result := 0.5 * (1 + T) + 0.5 * X * (1 - T * T) *
    K * (1 + 3 * 0.044715 * X * X);
end;

constructor TTransformerBlock.Create(const AEmbeddingDimension, AHeadCount,
  AFeedForwardDimension: Integer; var ARandom: TXorShift64Star);
begin
  inherited Create;
  if AFeedForwardDimension <= 0 then raise EDelphiLMTensorError.Create(
    'A dimensão do feed-forward deve ser positiva.');
  FEmbeddingDimension := AEmbeddingDimension; FFeedForwardDimension := AFeedForwardDimension;
  FLayerNorm1 := TLayerNorm.Create(FEmbeddingDimension);
  FLayerNorm2 := TLayerNorm.Create(FEmbeddingDimension);
  FAttention := TMultiHeadCausalSelfAttention.Create(FEmbeddingDimension, AHeadCount, ARandom);
  FFeedForwardInputWeights := TTrainableParameter.Create([FFeedForwardDimension, FEmbeddingDimension]);
  FFeedForwardInputBias := TTrainableParameter.Create([FFeedForwardDimension]);
  FFeedForwardOutputWeights := TTrainableParameter.Create([FEmbeddingDimension, FFeedForwardDimension]);
  FFeedForwardOutputBias := TTrainableParameter.Create([FEmbeddingDimension]);
  InitializeWeights(FFeedForwardInputWeights, FEmbeddingDimension, FFeedForwardDimension, ARandom);
  InitializeWeights(FFeedForwardOutputWeights, FFeedForwardDimension, FEmbeddingDimension, ARandom);
end;

destructor TTransformerBlock.Destroy;
begin
  ClearCache; FFeedForwardOutputBias.Free; FFeedForwardOutputWeights.Free;
  FFeedForwardInputBias.Free; FFeedForwardInputWeights.Free; FAttention.Free;
  FLayerNorm2.Free; FLayerNorm1.Free; inherited;
end;

procedure TTransformerBlock.ClearCache;
begin FreeAndNil(FCachedHiddenPre); FreeAndNil(FCachedNormalized2); FreeAndNil(FCachedResidual) end;

procedure TTransformerBlock.InitializeWeights(const AWeights: TTrainableParameter;
  const AFanIn, AFanOut: Integer; var ARandom: TXorShift64Star);
var I, O: Integer; Limit: Single;
begin
  Limit := Sqrt(6.0 / (AFanIn + AFanOut));
  for O := 0 to AWeights.Value.Dimension(0) - 1 do
    for I := 0 to AWeights.Value.Dimension(1) - 1 do
      AWeights.Value.SetValue((2 * ARandom.NextSingle - 1) * Limit, [O, I]);
end;

function TTransformerBlock.Forward(const AInput: TTensor): TTensor;
var A, N1: TTensor; C, H, P: Integer; Sum: Single;
begin
  if (AInput = nil) or (AInput.Rank <> 2) or
     (AInput.Dimension(1) <> FEmbeddingDimension) then
    raise EDelphiLMTensorError.CreateFmt('Bloco Transformer esperava shape [T,%d].', [FEmbeddingDimension]);
  ClearCache; N1 := FLayerNorm1.Forward(AInput); A := nil;
  try
    A := FAttention.Forward(N1);
    FCachedResidual := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
    for P := 0 to AInput.Dimension(0) - 1 do for C := 0 to FEmbeddingDimension - 1 do
      FCachedResidual.SetValue(AInput.ValueAt([P, C]) + A.ValueAt([P, C]), [P, C]);
    FCachedNormalized2 := FLayerNorm2.Forward(FCachedResidual);
    FCachedHiddenPre := TTensor.Create([AInput.Dimension(0), FFeedForwardDimension]);
    for P := 0 to AInput.Dimension(0) - 1 do for H := 0 to FFeedForwardDimension - 1 do
    begin
      Sum := FFeedForwardInputBias.Value.FlatValue(H);
      for C := 0 to FEmbeddingDimension - 1 do Sum := Sum +
        FFeedForwardInputWeights.Value.ValueAt([H, C]) * FCachedNormalized2.ValueAt([P, C]);
      FCachedHiddenPre.SetValue(Sum, [P, H]);
    end;
    Result := TTensor.Create([AInput.Dimension(0), FEmbeddingDimension]);
    for P := 0 to AInput.Dimension(0) - 1 do for C := 0 to FEmbeddingDimension - 1 do
    begin
      Sum := FFeedForwardOutputBias.Value.FlatValue(C);
      for H := 0 to FFeedForwardDimension - 1 do Sum := Sum +
        FFeedForwardOutputWeights.Value.ValueAt([C, H]) * GELU(FCachedHiddenPre.ValueAt([P, H]));
      Result.SetValue(FCachedResidual.ValueAt([P, C]) + Sum, [P, C]);
    end;
  finally A.Free; N1.Free end;
end;

function TTransformerBlock.Backward(const AGradOutput: TTensor): TTensor;
var C, H, P: Integer; G: Single; GR, GH, GN2, GA, GN1: TTensor;
begin
  if FCachedResidual = nil then raise EDelphiLMTensorError.Create('Backward do bloco exige Forward.');
  GR := TTensor.Create([AGradOutput.Dimension(0), FEmbeddingDimension]);
  GH := TTensor.Create([AGradOutput.Dimension(0), FFeedForwardDimension]);
  GN2 := nil; GA := nil; GN1 := nil;
  try
    for P := 0 to AGradOutput.Dimension(0) - 1 do for C := 0 to FEmbeddingDimension - 1 do
    begin
      G := AGradOutput.ValueAt([P, C]); GR.SetValue(G, [P, C]);
      FFeedForwardOutputBias.AddGradientFlat(C, G);
      for H := 0 to FFeedForwardDimension - 1 do
      begin
        FFeedForwardOutputWeights.Gradient.SetValue(
          FFeedForwardOutputWeights.Gradient.ValueAt([C, H]) +
          G * GELU(FCachedHiddenPre.ValueAt([P, H])), [C, H]);
        GH.SetValue(GH.ValueAt([P, H]) +
          FFeedForwardOutputWeights.Value.ValueAt([C, H]) * G, [P, H]);
      end;
    end;
    GN2 := TTensor.Create([AGradOutput.Dimension(0), FEmbeddingDimension]);
    for P := 0 to AGradOutput.Dimension(0) - 1 do for H := 0 to FFeedForwardDimension - 1 do
    begin
      G := GH.ValueAt([P, H]) * GELUDerivative(FCachedHiddenPre.ValueAt([P, H]));
      FFeedForwardInputBias.AddGradientFlat(H, G);
      for C := 0 to FEmbeddingDimension - 1 do
      begin
        FFeedForwardInputWeights.Gradient.SetValue(
          FFeedForwardInputWeights.Gradient.ValueAt([H, C]) +
          G * FCachedNormalized2.ValueAt([P, C]), [H, C]);
        GN2.SetValue(GN2.ValueAt([P, C]) +
          FFeedForwardInputWeights.Value.ValueAt([H, C]) * G, [P, C]);
      end;
    end;
    GA := FLayerNorm2.Backward(GN2);
    for P := 0 to GR.Dimension(0) - 1 do for C := 0 to FEmbeddingDimension - 1 do
      GR.SetValue(GR.ValueAt([P, C]) + GA.ValueAt([P, C]), [P, C]);
    FreeAndNil(GA); GA := FAttention.Backward(GR); GN1 := FLayerNorm1.Backward(GA);
    Result := TTensor.Create([GR.Dimension(0), FEmbeddingDimension]);
    for P := 0 to GR.Dimension(0) - 1 do for C := 0 to FEmbeddingDimension - 1 do
      Result.SetValue(GR.ValueAt([P, C]) + GN1.ValueAt([P, C]), [P, C]);
  finally GN1.Free; GA.Free; GN2.Free; GH.Free; GR.Free end;
end;

procedure TTransformerBlock.CollectParameters(const AList: TList<TTrainableParameter>);
begin
  FLayerNorm1.CollectParameters(AList); FAttention.CollectParameters(AList);
  FLayerNorm2.CollectParameters(AList); AList.Add(FFeedForwardInputWeights);
  AList.Add(FFeedForwardInputBias); AList.Add(FFeedForwardOutputWeights);
  AList.Add(FFeedForwardOutputBias);
end;

procedure TTransformerBlock.ZeroGrad;
begin
  FLayerNorm1.ZeroGrad; FAttention.ZeroGrad; FLayerNorm2.ZeroGrad;
  FFeedForwardInputWeights.ZeroGrad; FFeedForwardInputBias.ZeroGrad;
  FFeedForwardOutputWeights.ZeroGrad; FFeedForwardOutputBias.ZeroGrad;
end;

procedure TTransformerBlock.SetFeedForwardOutputWeight(const AOutput, AInput: Integer; const AValue: Single);
begin FFeedForwardOutputWeights.Value.SetValue(AValue, [AOutput, AInput]) end;

end.
