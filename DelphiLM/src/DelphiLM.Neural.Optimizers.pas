{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Neural.Optimizers;

interface

uses
  DelphiLM.Neural.Layers;

function LinearGradientGlobalNorm(const ALayer: TLinearLayer): Single;
function LinearGradientClipScale(const ALayer: TLinearLayer;
  const AMaximumNorm: Single): Single;
procedure SGDStep(const ALayer: TLinearLayer; const ALearningRate: Single;
  const AMaximumGradientNorm: Single = 0);

type
  TAdamOptimizer = class
  private
    FBeta1: Single;
    FBeta2: Single;
    FEpsilon: Single;
    FLayer: TLinearLayer;
    FLearningRate: Single;
    FFirstBiasMoment: TArray<Single>;
    FFirstWeightMoment: TArray<Single>;
    FSecondBiasMoment: TArray<Single>;
    FSecondWeightMoment: TArray<Single>;
    FStep: Integer;
    function WeightIndex(const AOutput, AInput: Integer): Integer;
  public
    constructor Create(const ALayer: TLinearLayer;
      const ALearningRate: Single = 1.0E-3;
      const ABeta1: Single = 0.9; const ABeta2: Single = 0.999;
      const AEpsilon: Single = 1.0E-8);
    procedure Step(const AMaximumGradientNorm: Single = 0);
  end;

implementation

uses
  System.Math,
  System.SysUtils;

function LinearGradientGlobalNorm(const ALayer: TLinearLayer): Single;
var
  Gradient: Double;
  InputIndex: Integer;
  OutputIndex: Integer;
  SumOfSquares: Double;
begin
  if ALayer = nil then
    raise EArgumentNilException.Create('ALayer');
  SumOfSquares := 0;
  for OutputIndex := 0 to ALayer.OutputSize - 1 do
  begin
    Gradient := ALayer.BiasGradientAt(OutputIndex);
    SumOfSquares := SumOfSquares + Gradient * Gradient;
    for InputIndex := 0 to ALayer.InputSize - 1 do
    begin
      Gradient := ALayer.WeightGradientAt(OutputIndex, InputIndex);
      SumOfSquares := SumOfSquares + Gradient * Gradient;
    end;
  end;
  Result := Sqrt(SumOfSquares);
end;

function LinearGradientClipScale(const ALayer: TLinearLayer;
  const AMaximumNorm: Single): Single;
var
  GradientNorm: Single;
begin
  if AMaximumNorm <= 0 then
    Exit(1);
  GradientNorm := LinearGradientGlobalNorm(ALayer);
  if GradientNorm > AMaximumNorm then
    Result := AMaximumNorm / GradientNorm
  else
    Result := 1;
end;

procedure SGDStep(const ALayer: TLinearLayer; const ALearningRate: Single;
  const AMaximumGradientNorm: Single);
var
  InputIndex: Integer;
  OutputIndex: Integer;
  Scale: Single;
begin
  if ALearningRate <= 0 then
    raise EArgumentOutOfRangeException.Create('ALearningRate');
  Scale := LinearGradientClipScale(ALayer, AMaximumGradientNorm);
  for OutputIndex := 0 to ALayer.OutputSize - 1 do
  begin
    ALayer.SetBias(OutputIndex, ALayer.BiasAt(OutputIndex) -
      ALearningRate * Scale * ALayer.BiasGradientAt(OutputIndex));
    for InputIndex := 0 to ALayer.InputSize - 1 do
      ALayer.SetWeight(OutputIndex, InputIndex,
        ALayer.WeightAt(OutputIndex, InputIndex) -
          ALearningRate * Scale *
          ALayer.WeightGradientAt(OutputIndex, InputIndex));
  end;
end;

constructor TAdamOptimizer.Create(const ALayer: TLinearLayer;
  const ALearningRate, ABeta1, ABeta2, AEpsilon: Single);
var
  WeightCount: Integer;
begin
  inherited Create;
  if ALayer = nil then
    raise EArgumentNilException.Create('ALayer');
  if ALearningRate <= 0 then
    raise EArgumentOutOfRangeException.Create('ALearningRate');
  if (ABeta1 <= 0) or (ABeta1 >= 1) then
    raise EArgumentOutOfRangeException.Create('ABeta1');
  if (ABeta2 <= 0) or (ABeta2 >= 1) then
    raise EArgumentOutOfRangeException.Create('ABeta2');
  if AEpsilon <= 0 then
    raise EArgumentOutOfRangeException.Create('AEpsilon');

  FLayer := ALayer;
  FLearningRate := ALearningRate;
  FBeta1 := ABeta1;
  FBeta2 := ABeta2;
  FEpsilon := AEpsilon;
  FStep := 0;
  WeightCount := FLayer.InputSize * FLayer.OutputSize;
  SetLength(FFirstWeightMoment, WeightCount);
  SetLength(FSecondWeightMoment, WeightCount);
  SetLength(FFirstBiasMoment, FLayer.OutputSize);
  SetLength(FSecondBiasMoment, FLayer.OutputSize);
end;

function TAdamOptimizer.WeightIndex(const AOutput,
  AInput: Integer): Integer;
begin
  Result := AOutput * FLayer.InputSize + AInput;
end;

procedure TAdamOptimizer.Step(const AMaximumGradientNorm: Single);
var
  BiasCorrection1: Double;
  BiasCorrection2: Double;
  Gradient: Single;
  Index: Integer;
  InputIndex: Integer;
  OutputIndex: Integer;
  Scale: Single;
  Update: Double;
begin
  Inc(FStep);
  Scale := LinearGradientClipScale(FLayer, AMaximumGradientNorm);
  BiasCorrection1 := 1 - Power(FBeta1, FStep);
  BiasCorrection2 := 1 - Power(FBeta2, FStep);

  for OutputIndex := 0 to FLayer.OutputSize - 1 do
  begin
    Gradient := Scale * FLayer.BiasGradientAt(OutputIndex);
    FFirstBiasMoment[OutputIndex] := FBeta1 *
      FFirstBiasMoment[OutputIndex] + (1 - FBeta1) * Gradient;
    FSecondBiasMoment[OutputIndex] := FBeta2 *
      FSecondBiasMoment[OutputIndex] +
      (1 - FBeta2) * Gradient * Gradient;
    Update := (FFirstBiasMoment[OutputIndex] / BiasCorrection1) /
      (Sqrt(FSecondBiasMoment[OutputIndex] / BiasCorrection2) + FEpsilon);
    FLayer.SetBias(OutputIndex,
      FLayer.BiasAt(OutputIndex) - FLearningRate * Update);

    for InputIndex := 0 to FLayer.InputSize - 1 do
    begin
      Index := WeightIndex(OutputIndex, InputIndex);
      Gradient := Scale *
        FLayer.WeightGradientAt(OutputIndex, InputIndex);
      FFirstWeightMoment[Index] := FBeta1 * FFirstWeightMoment[Index] +
        (1 - FBeta1) * Gradient;
      FSecondWeightMoment[Index] := FBeta2 * FSecondWeightMoment[Index] +
        (1 - FBeta2) * Gradient * Gradient;
      Update := (FFirstWeightMoment[Index] / BiasCorrection1) /
        (Sqrt(FSecondWeightMoment[Index] / BiasCorrection2) + FEpsilon);
      FLayer.SetWeight(OutputIndex, InputIndex,
        FLayer.WeightAt(OutputIndex, InputIndex) -
          FLearningRate * Update);
    end;
  end;
end;

end.
