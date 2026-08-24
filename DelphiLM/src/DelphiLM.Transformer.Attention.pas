{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Transformer.Attention;

interface

uses
  DelphiLM.Math.Tensor;

function CausalScaledDotProductAttention(const AQueries, AKeys,
  AValues: TTensor; out AWeights: TTensor): TTensor;

implementation

uses
  System.Math,
  System.SysUtils;

procedure ValidateAttentionInputs(const AQueries, AKeys,
  AValues: TTensor);
begin
  if (AQueries = nil) or (AKeys = nil) or (AValues = nil) then
    raise EArgumentNilException.Create('Q, K e V não podem ser nil.');
  if (AQueries.Rank <> 2) or (AKeys.Rank <> 2) or
     (AValues.Rank <> 2) then
    raise EDelphiLMTensorError.Create('Q, K e V devem ter rank 2.');
  if AQueries.Dimension(0) <> AKeys.Dimension(0) then
    raise EDelphiLMTensorError.Create(
      'Q e K devem ter o mesmo comprimento de sequência.');
  if AQueries.Dimension(0) <> AValues.Dimension(0) then
    raise EDelphiLMTensorError.Create(
      'Q e V devem ter o mesmo comprimento de sequência.');
  if AQueries.Dimension(1) <> AKeys.Dimension(1) then
    raise EDelphiLMTensorError.Create(
      'Q e K devem ter a mesma dimensão por cabeça.');
end;

function CausalScaledDotProductAttention(const AQueries, AKeys,
  AValues: TTensor; out AWeights: TTensor): TTensor;
var
  Component: Integer;
  Denominator: Double;
  KeyPosition: Integer;
  MaximumScore: Single;
  QueryPosition: Integer;
  Scale: Single;
  Score: Single;
  Scores: TArray<Single>;
  SequenceLength: Integer;
  Sum: Double;
begin
  ValidateAttentionInputs(AQueries, AKeys, AValues);
  SequenceLength := AQueries.Dimension(0);
  Scale := 1 / Sqrt(AQueries.Dimension(1));
  AWeights := TTensor.Create([SequenceLength, SequenceLength]);
  Result := TTensor.Create([SequenceLength, AValues.Dimension(1)]);
  SetLength(Scores, SequenceLength);

  for QueryPosition := 0 to SequenceLength - 1 do
  begin
    MaximumScore := -Infinity;
    for KeyPosition := 0 to QueryPosition do
    begin
      Sum := 0;
      for Component := 0 to AQueries.Dimension(1) - 1 do
        Sum := Sum +
          AQueries.ValueAt([QueryPosition, Component]) *
          AKeys.ValueAt([KeyPosition, Component]);
      Score := Sum * Scale;
      Scores[KeyPosition] := Score;
      if Score > MaximumScore then
        MaximumScore := Score;
    end;

    Denominator := 0;
    for KeyPosition := 0 to QueryPosition do
      Denominator := Denominator +
        Exp(Scores[KeyPosition] - MaximumScore);
    for KeyPosition := 0 to QueryPosition do
      AWeights.SetValue(
        Exp(Scores[KeyPosition] - MaximumScore) / Denominator,
        [QueryPosition, KeyPosition]);

    for Component := 0 to AValues.Dimension(1) - 1 do
    begin
      Sum := 0;
      for KeyPosition := 0 to QueryPosition do
        Sum := Sum +
          AWeights.ValueAt([QueryPosition, KeyPosition]) *
          AValues.ValueAt([KeyPosition, Component]);
      Result.SetValue(Sum, [QueryPosition, Component]);
    end;
  end;
end;

end.
