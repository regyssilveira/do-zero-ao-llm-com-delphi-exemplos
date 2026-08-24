{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Neural.Loss;

interface

uses
  DelphiLM.Math.Tensor;

function Softmax(const ALogits: TTensor): TTensor;
function LogSumExp(const ALogits: TTensor): Single;
function CrossEntropyFromLogits(const ALogits: TTensor;
  const ATargetToken: Integer): Single;

implementation

uses
  System.Math,
  System.SysUtils;

procedure ValidateLogits(const ALogits: TTensor);
begin
  if ALogits = nil then
    raise EArgumentNilException.Create('ALogits');
  if ALogits.Rank <> 1 then
    raise EDelphiLMTensorError.Create('Logits devem formar um vetor.');
  if ALogits.ElementCount < 2 then
    raise EDelphiLMTensorError.Create(
      'O vetor de logits precisa de ao menos duas classes.');
end;

function MaximumValue(const AValues: TTensor): Single;
var
  Index: Integer;
begin
  Result := AValues.FlatValue(0);
  for Index := 1 to AValues.ElementCount - 1 do
    if AValues.FlatValue(Index) > Result then
      Result := AValues.FlatValue(Index);
end;

function LogSumExp(const ALogits: TTensor): Single;
var
  Index: Integer;
  Maximum: Single;
  Sum: Double;
begin
  ValidateLogits(ALogits);
  Maximum := MaximumValue(ALogits);
  Sum := 0;
  for Index := 0 to ALogits.ElementCount - 1 do
    Sum := Sum + Exp(ALogits.FlatValue(Index) - Maximum);
  Result := Maximum + Ln(Sum);
end;

function Softmax(const ALogits: TTensor): TTensor;
var
  Denominator: Double;
  Index: Integer;
  Maximum: Single;
begin
  ValidateLogits(ALogits);
  Maximum := MaximumValue(ALogits);
  Denominator := 0;
  for Index := 0 to ALogits.ElementCount - 1 do
    Denominator := Denominator +
      Exp(ALogits.FlatValue(Index) - Maximum);

  Result := TTensor.Create([ALogits.ElementCount]);
  for Index := 0 to ALogits.ElementCount - 1 do
    Result.SetFlatValue(Index,
      Exp(ALogits.FlatValue(Index) - Maximum) / Denominator);
end;

function CrossEntropyFromLogits(const ALogits: TTensor;
  const ATargetToken: Integer): Single;
var
  Index: Integer;
  Maximum: Single;
  Sum: Double;
begin
  ValidateLogits(ALogits);
  if (ATargetToken < 0) or (ATargetToken >= ALogits.ElementCount) then
    raise EDelphiLMTensorError.CreateFmt(
      'Token-alvo %d fora do vocabulário de tamanho %d.',
      [ATargetToken, ALogits.ElementCount]);
  Maximum := MaximumValue(ALogits);
  Sum := 0;
  for Index := 0 to ALogits.ElementCount - 1 do
    Sum := Sum + Exp(ALogits.FlatValue(Index) - Maximum);
  Result := Single(Ln(Sum) + Double(Maximum) -
    Double(ALogits.FlatValue(ATargetToken)));
end;

end.
