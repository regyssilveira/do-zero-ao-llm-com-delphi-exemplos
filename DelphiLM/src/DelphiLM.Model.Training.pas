{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Model.Training;

interface

uses
  DelphiLM.Math.Tensor,
  DelphiLM.Model.LanguageModel,
  DelphiLM.Neural.Parameters,
  System.Generics.Collections;

type
  TTokenBatch = TArray<TArray<Integer>>;

  TModelAdamOptimizer = class
  private
    FBeta1, FBeta2, FEpsilon, FLearningRate: Single;
    FParameters: TList<TTrainableParameter>;
    FStep: Integer;
  public
    constructor Create(const AModel: TDelphiLanguageModel;
      const ALearningRate: Single = 3.0E-4;
      const ABeta1: Single = 0.9; const ABeta2: Single = 0.999;
      const AEpsilon: Single = 1.0E-8);
    destructor Destroy; override;
    function GradientGlobalNorm: Single;
    procedure Step(const AMaximumGradientNorm: Single = 1.0);
  end;

function NextTokenLossAndGradient(const ALogits: TTensor;
  const ATargets: TArray<Integer>; out AGradLogits: TTensor): Single;
function TrainStep(const AModel: TDelphiLanguageModel;
  const AOptimizer: TModelAdamOptimizer; const AInputs,
  ATargets: TArray<Integer>; const AMaximumGradientNorm: Single = 1.0): Single;
function TrainBatch(const AModel: TDelphiLanguageModel;
  const AOptimizer: TModelAdamOptimizer; const AInputs,
  ATargets: TTokenBatch; const AMaximumGradientNorm: Single = 1.0): Single;

implementation

uses
  System.Math, System.SysUtils;

constructor TModelAdamOptimizer.Create(const AModel: TDelphiLanguageModel;
  const ALearningRate, ABeta1, ABeta2, AEpsilon: Single);
begin
  inherited Create;
  if AModel = nil then raise EArgumentNilException.Create('AModel');
  if ALearningRate <= 0 then raise EArgumentOutOfRangeException.Create('ALearningRate');
  FParameters := TList<TTrainableParameter>.Create;
  AModel.CollectParameters(FParameters);
  FLearningRate := ALearningRate; FBeta1 := ABeta1; FBeta2 := ABeta2;
  FEpsilon := AEpsilon;
end;

destructor TModelAdamOptimizer.Destroy;
begin FParameters.Free; inherited end;

function TModelAdamOptimizer.GradientGlobalNorm: Single;
var P: TTrainableParameter; Sum: Double;
begin
  Sum := 0;
  for P in FParameters do Sum := Sum + P.GradientSquaredSum;
  Result := Sqrt(Sum);
end;

procedure TModelAdamOptimizer.Step(const AMaximumGradientNorm: Single);
var BC1, BC2, Norm, Scale: Single; P: TTrainableParameter;
begin
  Inc(FStep); Norm := GradientGlobalNorm; Scale := 1;
  if (AMaximumGradientNorm > 0) and (Norm > AMaximumGradientNorm) then
    Scale := AMaximumGradientNorm / Norm;
  BC1 := 1 - Power(FBeta1, FStep); BC2 := 1 - Power(FBeta2, FStep);
  for P in FParameters do
    P.AdamStep(FLearningRate, FBeta1, FBeta2, FEpsilon, Scale, BC1, BC2);
end;

function NextTokenLossAndGradient(const ALogits: TTensor;
  const ATargets: TArray<Integer>; out AGradLogits: TTensor): Single;
var Denominator, MaxLogit, Probability, SumExp: Double;
  P, T, V: Integer;
begin
  if (ALogits = nil) or (ALogits.Rank <> 2) or
     (ALogits.Dimension(0) <> Length(ATargets)) then
    raise EDelphiLMTensorError.Create('Targets incompatíveis com os logits.');
  AGradLogits := TTensor.Create([ALogits.Dimension(0), ALogits.Dimension(1)]);
  Result := 0; Denominator := ALogits.Dimension(0);
  for P := 0 to ALogits.Dimension(0) - 1 do
  begin
    T := ATargets[P];
    if (T < 0) or (T >= ALogits.Dimension(1)) then
      raise EDelphiLMTensorError.CreateFmt('Target %d fora do vocabulário.', [T]);
    MaxLogit := ALogits.ValueAt([P, 0]);
    for V := 1 to ALogits.Dimension(1) - 1 do
      if ALogits.ValueAt([P, V]) > MaxLogit then MaxLogit := ALogits.ValueAt([P, V]);
    SumExp := 0;
    for V := 0 to ALogits.Dimension(1) - 1 do
      SumExp := SumExp + Exp(ALogits.ValueAt([P, V]) - MaxLogit);
    Result := Result + (MaxLogit + Ln(SumExp) - ALogits.ValueAt([P, T])) / Denominator;
    for V := 0 to ALogits.Dimension(1) - 1 do
    begin
      Probability := Exp(ALogits.ValueAt([P, V]) - MaxLogit) / SumExp;
      if V = T then Probability := Probability - 1;
      AGradLogits.SetValue(Probability / Denominator, [P, V]);
    end;
  end;
end;

function TrainStep(const AModel: TDelphiLanguageModel;
  const AOptimizer: TModelAdamOptimizer; const AInputs,
  ATargets: TArray<Integer>; const AMaximumGradientNorm: Single): Single;
var GradLogits, Logits: TTensor;
begin
  AModel.ZeroGrad; Logits := AModel.Forward(AInputs); GradLogits := nil;
  try
    Result := NextTokenLossAndGradient(Logits, ATargets, GradLogits);
    AModel.Backward(GradLogits); AOptimizer.Step(AMaximumGradientNorm);
  finally GradLogits.Free; Logits.Free end;
end;

function TrainBatch(const AModel: TDelphiLanguageModel;
  const AOptimizer: TModelAdamOptimizer; const AInputs,
  ATargets: TTokenBatch; const AMaximumGradientNorm: Single): Single;
var
  Example: Integer;
  GradLogits: TTensor;
  Index: Integer;
  Logits: TTensor;
begin
  if (Length(AInputs) = 0) or (Length(AInputs) <> Length(ATargets)) then
    raise EDelphiLMTensorError.Create('Batch de entradas e targets incompatível.');
  AModel.ZeroGrad;
  Result := 0;
  for Example := 0 to High(AInputs) do
  begin
    Logits := AModel.Forward(AInputs[Example]);
    GradLogits := nil;
    try
      Result := Result + NextTokenLossAndGradient(
        Logits, ATargets[Example], GradLogits) / Length(AInputs);
      for Index := 0 to GradLogits.ElementCount - 1 do
        GradLogits.SetFlatValue(Index,
          GradLogits.FlatValue(Index) / Length(AInputs));
      AModel.Backward(GradLogits);
    finally
      GradLogits.Free;
      Logits.Free;
    end;
  end;
  AOptimizer.Step(AMaximumGradientNorm);
end;

end.
