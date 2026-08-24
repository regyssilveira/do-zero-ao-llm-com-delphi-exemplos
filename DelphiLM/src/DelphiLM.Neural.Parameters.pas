{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Neural.Parameters;

interface

uses
  DelphiLM.Math.Tensor;

type
  TTrainableParameter = class
  private
    FFirstMoment: TTensor;
    FGradient: TTensor;
    FSecondMoment: TTensor;
    FValue: TTensor;
  public
    constructor Create(const AShape: array of Integer);
    destructor Destroy; override;
    procedure AddGradientFlat(const AIndex: Integer; const AValue: Single);
    procedure AdamStep(const ALearningRate, ABeta1, ABeta2,
      AEpsilon, AGradientScale, ABiasCorrection1,
      ABiasCorrection2: Single);
    function GradientSquaredSum: Double;
    procedure ZeroGrad;
    property Gradient: TTensor read FGradient;
    property Value: TTensor read FValue;
  end;

implementation

uses
  System.Math;

constructor TTrainableParameter.Create(const AShape: array of Integer);
begin
  inherited Create;
  FValue := TTensor.Create(AShape);
  FGradient := TTensor.Create(AShape);
  FFirstMoment := TTensor.Create(AShape);
  FSecondMoment := TTensor.Create(AShape);
end;

destructor TTrainableParameter.Destroy;
begin
  FSecondMoment.Free;
  FFirstMoment.Free;
  FGradient.Free;
  FValue.Free;
  inherited;
end;

procedure TTrainableParameter.AddGradientFlat(const AIndex: Integer;
  const AValue: Single);
begin
  FGradient.SetFlatValue(AIndex,
    FGradient.FlatValue(AIndex) + AValue);
end;

function TTrainableParameter.GradientSquaredSum: Double;
var
  GradientValue: Double;
  Index: Integer;
begin
  Result := 0;
  for Index := 0 to FGradient.ElementCount - 1 do
  begin
    GradientValue := FGradient.FlatValue(Index);
    Result := Result + GradientValue * GradientValue;
  end;
end;

procedure TTrainableParameter.AdamStep(const ALearningRate, ABeta1,
  ABeta2, AEpsilon, AGradientScale, ABiasCorrection1,
  ABiasCorrection2: Single);
var
  FirstMoment: Double;
  GradientValue: Double;
  Index: Integer;
  SecondMoment: Double;
  Update: Double;
begin
  for Index := 0 to FValue.ElementCount - 1 do
  begin
    GradientValue := FGradient.FlatValue(Index) * AGradientScale;
    FirstMoment := ABeta1 * FFirstMoment.FlatValue(Index) +
      (1 - ABeta1) * GradientValue;
    SecondMoment := ABeta2 * FSecondMoment.FlatValue(Index) +
      (1 - ABeta2) * GradientValue * GradientValue;
    FFirstMoment.SetFlatValue(Index, FirstMoment);
    FSecondMoment.SetFlatValue(Index, SecondMoment);
    Update := (FirstMoment / ABiasCorrection1) /
      (Sqrt(SecondMoment / ABiasCorrection2) + AEpsilon);
    FValue.SetFlatValue(Index,
      FValue.FlatValue(Index) - ALearningRate * Update);
  end;
end;

procedure TTrainableParameter.ZeroGrad;
begin
  FGradient.Fill(0);
end;

end.
