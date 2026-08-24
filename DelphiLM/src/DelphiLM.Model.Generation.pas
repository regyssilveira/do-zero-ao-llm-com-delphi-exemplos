{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Model.Generation;

interface

uses
  DelphiLM.Core.Random, DelphiLM.Math.Tensor,
  DelphiLM.Model.LanguageModel;

function SampleNextToken(const ALogits: TTensor; const APosition: Integer;
  const ATemperature: Single; const ATopK: Integer;
  var ARandom: TXorShift64Star): Integer;
function GenerateTokens(const AModel: TDelphiLanguageModel;
  const APrompt: TArray<Integer>; const AMaxNewTokens: Integer;
  const ATemperature: Single; const ATopK: Integer;
  var ARandom: TXorShift64Star): TArray<Integer>;

implementation

uses
  System.Math, System.SysUtils;

function SampleNextToken(const ALogits: TTensor; const APosition: Integer;
  const ATemperature: Single; const ATopK: Integer;
  var ARandom: TXorShift64Star): Integer;
var
  Candidates: TArray<Integer>;
  CandidateCount, C, K, Token, BestToken: Integer;
  BestValue, MaxScaled, Sum, Draw, Scaled: Double;
  Used: TArray<Boolean>;
begin
  if (ALogits = nil) or (ALogits.Rank <> 2) then
    raise EDelphiLMTensorError.Create('Amostragem exige logits [T,V].');
  if (APosition < 0) or (APosition >= ALogits.Dimension(0)) then
    raise ERangeError.Create('Posição de amostragem inválida.');
  if ATemperature <= 0 then
    raise EArgumentOutOfRangeException.Create('ATemperature');
  CandidateCount := ALogits.Dimension(1);
  if (ATopK > 0) and (ATopK < CandidateCount) then CandidateCount := ATopK;
  SetLength(Candidates, CandidateCount);
  SetLength(Used, ALogits.Dimension(1));
  for K := 0 to CandidateCount - 1 do
  begin
    BestToken := -1; BestValue := -Infinity;
    for Token := 0 to ALogits.Dimension(1) - 1 do
      if (not Used[Token]) and (ALogits.ValueAt([APosition, Token]) > BestValue) then
      begin BestValue := ALogits.ValueAt([APosition, Token]); BestToken := Token end;
    Candidates[K] := BestToken; Used[BestToken] := True;
  end;
  MaxScaled := ALogits.ValueAt([APosition, Candidates[0]]) / ATemperature;
  Sum := 0;
  for C := 0 to High(Candidates) do
    Sum := Sum + Exp(ALogits.ValueAt([APosition, Candidates[C]]) /
      ATemperature - MaxScaled);
  Draw := ARandom.NextSingle * Sum;
  for C := 0 to High(Candidates) do
  begin
    Scaled := Exp(ALogits.ValueAt([APosition, Candidates[C]]) /
      ATemperature - MaxScaled);
    Draw := Draw - Scaled;
    if Draw <= 0 then Exit(Candidates[C]);
  end;
  Result := Candidates[High(Candidates)];
end;

function GenerateTokens(const AModel: TDelphiLanguageModel;
  const APrompt: TArray<Integer>; const AMaxNewTokens: Integer;
  const ATemperature: Single; const ATopK: Integer;
  var ARandom: TXorShift64Star): TArray<Integer>;
var
  Context: TArray<Integer>;
  ContextStart, I, NextToken, Step: Integer;
  Logits: TTensor;
begin
  if AModel = nil then raise EArgumentNilException.Create('AModel');
  if Length(APrompt) = 0 then raise EDelphiLMTensorError.Create('Prompt não pode ser vazio.');
  if AMaxNewTokens < 0 then raise EArgumentOutOfRangeException.Create('AMaxNewTokens');
  Result := Copy(APrompt);
  for Step := 1 to AMaxNewTokens do
  begin
    ContextStart := Max(0, Length(Result) - AModel.Config.ContextLength);
    SetLength(Context, Length(Result) - ContextStart);
    for I := 0 to High(Context) do Context[I] := Result[ContextStart + I];
    Logits := AModel.Forward(Context);
    try
      NextToken := SampleNextToken(Logits, High(Context),
        ATemperature, ATopK, ARandom);
    finally Logits.Free end;
    SetLength(Result, Length(Result) + 1);
    Result[High(Result)] := NextToken;
  end;
end;

end.
