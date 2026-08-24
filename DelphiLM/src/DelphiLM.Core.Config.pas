{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Core.Config;

interface

uses
  System.SysUtils;

type
  EDelphiLMConfigError = class(Exception);

  TDelphiLMConfig = record
  public
    VocabularySize: Integer;
    ContextLength: Integer;
    EmbeddingDimension: Integer;
    BlockCount: Integer;
    AttentionHeadCount: Integer;
    FeedForwardDimension: Integer;
    BatchSize: Integer;
    LearningRate: Single;
    Seed: UInt64;
    class function Default: TDelphiLMConfig; static;
    procedure Validate;
    function ParameterCount: Int64;
  end;

implementation

class function TDelphiLMConfig.Default: TDelphiLMConfig;
begin
  Result.VocabularySize := 128;
  Result.ContextLength := 64;
  Result.EmbeddingDimension := 64;
  Result.BlockCount := 2;
  Result.AttentionHeadCount := 4;
  Result.FeedForwardDimension := 256;
  Result.BatchSize := 8;
  Result.LearningRate := 3.0E-4;
  Result.Seed := 20260824;
end;

procedure TDelphiLMConfig.Validate;
begin
  if VocabularySize <= 1 then
    raise EDelphiLMConfigError.Create('VocabularySize deve ser maior que 1.');
  if ContextLength <= 0 then
    raise EDelphiLMConfigError.Create('ContextLength deve ser positivo.');
  if EmbeddingDimension <= 0 then
    raise EDelphiLMConfigError.Create('EmbeddingDimension deve ser positivo.');
  if BlockCount <= 0 then
    raise EDelphiLMConfigError.Create('BlockCount deve ser positivo.');
  if AttentionHeadCount <= 0 then
    raise EDelphiLMConfigError.Create('AttentionHeadCount deve ser positivo.');
  if (EmbeddingDimension mod AttentionHeadCount) <> 0 then
    raise EDelphiLMConfigError.Create(
      'EmbeddingDimension deve ser divisível por AttentionHeadCount.');
  if FeedForwardDimension <= 0 then
    raise EDelphiLMConfigError.Create('FeedForwardDimension deve ser positivo.');
  if BatchSize <= 0 then
    raise EDelphiLMConfigError.Create('BatchSize deve ser positivo.');
  if LearningRate <= 0 then
    raise EDelphiLMConfigError.Create('LearningRate deve ser positivo.');
end;

function TDelphiLMConfig.ParameterCount: Int64;
var
  PerBlock: Int64;
begin
  Validate;

  { Duas LayerNorms, Q/K/V, projeção da atenção e MLP com biases. }
  PerBlock :=
    (4 * EmbeddingDimension) +
    (3 * (Int64(EmbeddingDimension) * EmbeddingDimension +
      EmbeddingDimension)) +
    (Int64(EmbeddingDimension) * EmbeddingDimension + EmbeddingDimension) +
    (Int64(EmbeddingDimension) * FeedForwardDimension +
      FeedForwardDimension) +
    (Int64(FeedForwardDimension) * EmbeddingDimension +
      EmbeddingDimension);

  Result :=
    Int64(VocabularySize) * EmbeddingDimension +
    Int64(ContextLength) * EmbeddingDimension +
    Int64(BlockCount) * PerBlock +
    (2 * EmbeddingDimension) +
    VocabularySize;
end;

end.
