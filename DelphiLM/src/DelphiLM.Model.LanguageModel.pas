{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Model.LanguageModel;

interface

uses
  DelphiLM.Core.Config,
  DelphiLM.Math.Tensor,
  DelphiLM.Transformer.Block,
  DelphiLM.Transformer.Embeddings;

type
  TDelphiLanguageModel = class
  private
    FBlocks: TArray<TTransformerBlock>;
    FConfig: TDelphiLMConfig;
    FEmbeddings: TTokenPositionEmbedding;
    FFinalLayerNorm: TLayerNorm;
    FOutputBias: TTensor;
  public
    constructor Create(const AConfig: TDelphiLMConfig);
    destructor Destroy; override;
    function BlockAt(const AIndex: Integer): TTransformerBlock;
    function Forward(const ATokens: TArray<Integer>): TTensor;
    procedure SetOutputBias(const AToken: Integer; const AValue: Single);
    property Config: TDelphiLMConfig read FConfig;
    property Embeddings: TTokenPositionEmbedding read FEmbeddings;
  end;

implementation

uses
  DelphiLM.Core.Random,
  System.SysUtils;

constructor TDelphiLanguageModel.Create(const AConfig: TDelphiLMConfig);
var
  BlockIndex: Integer;
  Random: TXorShift64Star;
begin
  inherited Create;
  AConfig.Validate;
  FConfig := AConfig;
  Random.Initialize(FConfig.Seed);
  FEmbeddings := TTokenPositionEmbedding.Create(
    FConfig.VocabularySize, FConfig.ContextLength,
    FConfig.EmbeddingDimension, Random);
  SetLength(FBlocks, FConfig.BlockCount);
  for BlockIndex := 0 to High(FBlocks) do
    FBlocks[BlockIndex] := TTransformerBlock.Create(
      FConfig.EmbeddingDimension, FConfig.AttentionHeadCount,
      FConfig.FeedForwardDimension, Random);
  FFinalLayerNorm := TLayerNorm.Create(FConfig.EmbeddingDimension);
  FOutputBias := TTensor.Create([FConfig.VocabularySize]);
end;

destructor TDelphiLanguageModel.Destroy;
var
  BlockIndex: Integer;
begin
  FOutputBias.Free;
  FFinalLayerNorm.Free;
  for BlockIndex := High(FBlocks) downto 0 do
    FBlocks[BlockIndex].Free;
  FEmbeddings.Free;
  inherited;
end;

function TDelphiLanguageModel.BlockAt(
  const AIndex: Integer): TTransformerBlock;
begin
  if (AIndex < 0) or (AIndex > High(FBlocks)) then
    raise ERangeError.CreateFmt('Bloco %d fora do modelo.', [AIndex]);
  Result := FBlocks[AIndex];
end;

function TDelphiLanguageModel.Forward(
  const ATokens: TArray<Integer>): TTensor;
var
  BlockIndex: Integer;
  Component: Integer;
  Hidden: TTensor;
  NextHidden: TTensor;
  Position: Integer;
  Sum: Double;
  Token: Integer;
begin
  Hidden := FEmbeddings.Forward(ATokens);
  try
    for BlockIndex := 0 to High(FBlocks) do
    begin
      NextHidden := FBlocks[BlockIndex].Forward(Hidden);
      Hidden.Free;
      Hidden := NextHidden;
    end;
    NextHidden := FFinalLayerNorm.Forward(Hidden);
    Hidden.Free;
    Hidden := NextHidden;

    Result := TTensor.Create([Length(ATokens), FConfig.VocabularySize]);
    for Position := 0 to High(ATokens) do
      for Token := 0 to FConfig.VocabularySize - 1 do
      begin
        Sum := FOutputBias.FlatValue(Token);
        for Component := 0 to FConfig.EmbeddingDimension - 1 do
          Sum := Sum + Hidden.ValueAt([Position, Component]) *
            FEmbeddings.TokenTable.ValueAt(Token, Component);
        Result.SetValue(Sum, [Position, Token]);
      end;
  finally
    Hidden.Free;
  end;
end;

procedure TDelphiLanguageModel.SetOutputBias(
  const AToken: Integer; const AValue: Single);
begin
  FOutputBias.SetFlatValue(AToken, AValue);
end;

end.
