{ Licensed under the Apache License, Version 2.0. See LICENSE. }
unit DelphiLM.Model.LanguageModel;

interface

uses
  DelphiLM.Core.Config,
  DelphiLM.Math.Tensor,
  DelphiLM.Neural.Parameters,
  DelphiLM.Transformer.Block,
  DelphiLM.Transformer.Embeddings,
  System.Generics.Collections;

type
  TDelphiLanguageModel = class
  private
    FBlocks: TArray<TTransformerBlock>;
    FCachedFinalHidden: TTensor;
    FCachedTokens: TArray<Integer>;
    FConfig: TDelphiLMConfig;
    FEmbeddings: TTokenPositionEmbedding;
    FFinalLayerNorm: TLayerNorm;
    FOutputBias: TTrainableParameter;
  public
    constructor Create(const AConfig: TDelphiLMConfig);
    destructor Destroy; override;
    function BlockAt(const AIndex: Integer): TTransformerBlock;
    function Forward(const ATokens: TArray<Integer>): TTensor;
    procedure Backward(const AGradLogits: TTensor);
    procedure CollectParameters(const AList: TList<TTrainableParameter>);
    procedure ZeroGrad;
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
  FOutputBias := TTrainableParameter.Create([FConfig.VocabularySize]);
end;

destructor TDelphiLanguageModel.Destroy;
var
  BlockIndex: Integer;
begin
  FCachedFinalHidden.Free;
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
  FCachedFinalHidden.Free;
  FCachedFinalHidden := nil;
  FCachedTokens := Copy(ATokens);
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

    FCachedFinalHidden := TTensor.Create([
      Hidden.Dimension(0), Hidden.Dimension(1)]);
    for Component := 0 to Hidden.ElementCount - 1 do
      FCachedFinalHidden.SetFlatValue(Component, Hidden.FlatValue(Component));
    Result := TTensor.Create([Length(ATokens), FConfig.VocabularySize]);
    for Position := 0 to High(ATokens) do
      for Token := 0 to FConfig.VocabularySize - 1 do
      begin
        Sum := FOutputBias.Value.FlatValue(Token);
        for Component := 0 to FConfig.EmbeddingDimension - 1 do
          Sum := Sum + Hidden.ValueAt([Position, Component]) *
            FEmbeddings.TokenTable.ValueAt(Token, Component);
        Result.SetValue(Sum, [Position, Token]);
      end;
  finally
    Hidden.Free;
  end;
end;

procedure TDelphiLanguageModel.Backward(const AGradLogits: TTensor);
var
  BlockIndex: Integer;
  Component: Integer;
  GradHidden: TTensor;
  NextGradient: TTensor;
  Position: Integer;
  Token: Integer;
begin
  if FCachedFinalHidden = nil then
    raise EDelphiLMTensorError.Create('Backward exige um Forward anterior.');
  if (AGradLogits = nil) or (AGradLogits.Rank <> 2) or
     (AGradLogits.Dimension(0) <> Length(FCachedTokens)) or
     (AGradLogits.Dimension(1) <> FConfig.VocabularySize) then
    raise EDelphiLMTensorError.Create('Gradiente incompatível com os logits.');
  GradHidden := TTensor.Create([
    Length(FCachedTokens), FConfig.EmbeddingDimension]);
  try
    for Position := 0 to High(FCachedTokens) do
      for Token := 0 to FConfig.VocabularySize - 1 do
      begin
        FOutputBias.AddGradientFlat(Token,
          AGradLogits.ValueAt([Position, Token]));
        for Component := 0 to FConfig.EmbeddingDimension - 1 do
        begin
          GradHidden.SetValue(
            GradHidden.ValueAt([Position, Component]) +
            AGradLogits.ValueAt([Position, Token]) *
            FEmbeddings.TokenTable.ValueAt(Token, Component),
            [Position, Component]);
          FEmbeddings.TokenTable.AddGradient(Token, Component,
            AGradLogits.ValueAt([Position, Token]) *
            FCachedFinalHidden.ValueAt([Position, Component]));
        end;
      end;
    NextGradient := FFinalLayerNorm.Backward(GradHidden);
    GradHidden.Free;
    GradHidden := NextGradient;
    for BlockIndex := High(FBlocks) downto 0 do
    begin
      NextGradient := FBlocks[BlockIndex].Backward(GradHidden);
      GradHidden.Free;
      GradHidden := NextGradient;
    end;
    FEmbeddings.Backward(GradHidden);
  finally
    GradHidden.Free;
  end;
end;

procedure TDelphiLanguageModel.CollectParameters(
  const AList: TList<TTrainableParameter>);
var
  BlockIndex: Integer;
begin
  FEmbeddings.CollectParameters(AList);
  for BlockIndex := 0 to High(FBlocks) do
    FBlocks[BlockIndex].CollectParameters(AList);
  FFinalLayerNorm.CollectParameters(AList);
  AList.Add(FOutputBias);
end;

procedure TDelphiLanguageModel.ZeroGrad;
var
  BlockIndex: Integer;
begin
  FEmbeddings.ZeroGrad;
  for BlockIndex := 0 to High(FBlocks) do
    FBlocks[BlockIndex].ZeroGrad;
  FFinalLayerNorm.ZeroGrad;
  FOutputBias.ZeroGrad;
end;

procedure TDelphiLanguageModel.SetOutputBias(
  const AToken: Integer; const AValue: Single);
begin
  FOutputBias.Value.SetFlatValue(AToken, AValue);
end;

end.
