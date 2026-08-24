{ Licensed under the Apache License, Version 2.0. See LICENSE. }
program DelphiLM_Tests;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  DelphiLM.Core.Config in '..\src\DelphiLM.Core.Config.pas',
  DelphiLM.Core.Random in '..\src\DelphiLM.Core.Random.pas',
  DelphiLM.Data.Tokenizer in '..\src\DelphiLM.Data.Tokenizer.pas',
  DelphiLM.Math.Tensor in '..\src\DelphiLM.Math.Tensor.pas',
  DelphiLM.Neural.Layers in '..\src\DelphiLM.Neural.Layers.pas',
  DelphiLM.Neural.Loss in '..\src\DelphiLM.Neural.Loss.pas',
  DelphiLM.Neural.Optimizers in '..\src\DelphiLM.Neural.Optimizers.pas',
  DelphiLM.Transformer.Embeddings in '..\src\DelphiLM.Transformer.Embeddings.pas',
  DelphiLM.Transformer.Attention in '..\src\DelphiLM.Transformer.Attention.pas',
  DelphiLM.Transformer.MultiHeadAttention in '..\src\DelphiLM.Transformer.MultiHeadAttention.pas',
  DelphiLM.Transformer.Block in '..\src\DelphiLM.Transformer.Block.pas',
  DelphiLM.Model.LanguageModel in '..\src\DelphiLM.Model.LanguageModel.pas',
  DelphiLM.Neural.Parameters in '..\src\DelphiLM.Neural.Parameters.pas',
  DelphiLM.Model.Training in '..\src\DelphiLM.Model.Training.pas',
  DelphiLM.Model.Generation in '..\src\DelphiLM.Model.Generation.pas',
  DelphiLM.Model.Checkpoint in '..\src\DelphiLM.Model.Checkpoint.pas',
  System.IOUtils;

procedure AssertTrue(const ACondition: Boolean; const AMessage: string);
begin
  if not ACondition then
    raise Exception.Create(AMessage);
end;

procedure TestDefaultConfig;
var
  Config: TDelphiLMConfig;
begin
  Config := TDelphiLMConfig.Default;
  Config.Validate;
  AssertTrue(Config.ParameterCount = 112512,
    'A configuração-base deve ter 112.512 parâmetros.');
end;

procedure TestInvalidHeadCount;
var
  Config: TDelphiLMConfig;
  RaisedExpectedError: Boolean;
begin
  Config := TDelphiLMConfig.Default;
  Config.AttentionHeadCount := 3;
  RaisedExpectedError := False;
  try
    Config.Validate;
  except
    on E: EDelphiLMConfigError do
      RaisedExpectedError := True;
  end;
  AssertTrue(RaisedExpectedError,
    'Configuração incompatível entre embedding e heads deve falhar.');
end;

procedure TestDeterministicRandom;
var
  LeftRandom: TXorShift64Star;
  RightRandom: TXorShift64Star;
  Index: Integer;
begin
  LeftRandom.Initialize(42);
  RightRandom.Initialize(42);
  for Index := 1 to 16 do
    AssertTrue(LeftRandom.NextUInt64 = RightRandom.NextUInt64,
      'A mesma seed deve reproduzir a mesma sequência.');
end;

procedure TestTokenizerRoundTrip;
const
  Corpus = 'Delphi também aprende.';
var
  Tokenizer: TCharacterTokenizer;
  Tokens: TArray<Integer>;
begin
  Tokenizer := TCharacterTokenizer.Create(Corpus);
  try
    Tokens := Tokenizer.Encode(Corpus);
    AssertTrue(Tokenizer.Decode(Tokens) = Corpus,
      'Encode seguido de Decode deve recuperar o texto normalizado.');
  finally
    Tokenizer.Free;
  end;
end;

procedure TestTokenizerNormalizesNFC;
var
  Composed: string;
  Decomposed: string;
  Tokenizer: TCharacterTokenizer;
begin
  Composed := 'a' + Char($00E7) + Char($00E3) + 'o';
  Decomposed := 'ac' + Char($0327) + Char($00E3) + 'o';
  Tokenizer := TCharacterTokenizer.Create(Composed);
  try
    AssertTrue(Tokenizer.Decode(Tokenizer.Encode(Decomposed)) = Composed,
      'Formas Unicode equivalentes devem ser normalizadas para NFC.');
  finally
    Tokenizer.Free;
  end;
end;

procedure TestTensorRowMajor;
var
  Tensor: TTensor;
begin
  Tensor := TTensor.Create([2, 3]);
  try
    Tensor.SetValue(7.5, [1, 2]);
    AssertTrue(Abs(Tensor.FlatValue(5) - 7.5) < 1.0E-6,
      'O índice [1, 2] deve ocupar a posição linear 5.');
  finally
    Tensor.Free;
  end;
end;

procedure TestDotProduct;
var
  LeftVector: TTensor;
  RightVector: TTensor;
begin
  LeftVector := TTensor.Create([3]);
  RightVector := TTensor.Create([3]);
  try
    LeftVector.SetFlatValue(0, 1);
    LeftVector.SetFlatValue(1, 2);
    LeftVector.SetFlatValue(2, 3);
    RightVector.SetFlatValue(0, 4);
    RightVector.SetFlatValue(1, 5);
    RightVector.SetFlatValue(2, 6);
    AssertTrue(Abs(DotProduct(LeftVector, RightVector) - 32) < 1.0E-6,
      'O produto escalar de [1,2,3] e [4,5,6] deve ser 32.');
  finally
    RightVector.Free;
    LeftVector.Free;
  end;
end;

procedure TestMatrixMultiply;
var
  LeftMatrix: TTensor;
  Product: TTensor;
  RightMatrix: TTensor;
begin
  LeftMatrix := TTensor.Create([2, 2]);
  RightMatrix := TTensor.Create([2, 2]);
  Product := nil;
  try
    LeftMatrix.SetValue(1, [0, 0]);
    LeftMatrix.SetValue(2, [0, 1]);
    LeftMatrix.SetValue(3, [1, 0]);
    LeftMatrix.SetValue(4, [1, 1]);
    RightMatrix.SetValue(5, [0, 0]);
    RightMatrix.SetValue(6, [0, 1]);
    RightMatrix.SetValue(7, [1, 0]);
    RightMatrix.SetValue(8, [1, 1]);
    Product := MatrixMultiply(LeftMatrix, RightMatrix);
    AssertTrue(Abs(Product.ValueAt([0, 0]) - 19) < 1.0E-6,
      'O primeiro elemento do produto deve ser 19.');
    AssertTrue(Abs(Product.ValueAt([1, 1]) - 50) < 1.0E-6,
      'O último elemento do produto deve ser 50.');
  finally
    Product.Free;
    RightMatrix.Free;
    LeftMatrix.Free;
  end;
end;

procedure TestLinearForward;
var
  Input: TTensor;
  Layer: TLinearLayer;
  Output: TTensor;
  Random: TXorShift64Star;
begin
  Random.Initialize(42);
  Layer := TLinearLayer.Create(2, 2, Random);
  Input := TTensor.Create([2]);
  Output := nil;
  try
    Layer.SetWeight(0, 0, 0.5);
    Layer.SetWeight(0, 1, -1.0);
    Layer.SetBias(0, 0.1);
    Layer.SetWeight(1, 0, 2.0);
    Layer.SetWeight(1, 1, 0.25);
    Layer.SetBias(1, -0.2);
    Input.SetFlatValue(0, 2.0);
    Input.SetFlatValue(1, 3.0);
    Output := Layer.Forward(Input);
    AssertTrue(Abs(Output.FlatValue(0) - (-1.9)) < 1.0E-5,
      'A primeira saída linear deve ser -1,9.');
    AssertTrue(Abs(Output.FlatValue(1) - 4.55) < 1.0E-5,
      'A segunda saída linear deve ser 4,55.');
  finally
    Output.Free;
    Input.Free;
    Layer.Free;
  end;
end;

procedure TestActivations;
begin
  AssertTrue(ReLU(-2) = 0, 'ReLU deve zerar entrada negativa.');
  AssertTrue(ReLU(2) = 2, 'ReLU deve preservar entrada positiva.');
  AssertTrue(Abs(GELU(0)) < 1.0E-6, 'GELU de zero deve ser zero.');
  AssertTrue(GELU(1) > 0, 'GELU de um deve ser positiva.');
end;

procedure TestLinearInitializationIsDeterministic;
var
  LeftLayer: TLinearLayer;
  LeftRandom: TXorShift64Star;
  RightLayer: TLinearLayer;
  RightRandom: TXorShift64Star;
begin
  LeftRandom.Initialize(123);
  RightRandom.Initialize(123);
  LeftLayer := TLinearLayer.Create(3, 2, LeftRandom);
  RightLayer := TLinearLayer.Create(3, 2, RightRandom);
  try
    AssertTrue(LeftLayer.WeightAt(1, 2) = RightLayer.WeightAt(1, 2),
      'Mesma seed deve produzir os mesmos pesos iniciais.');
  finally
    RightLayer.Free;
    LeftLayer.Free;
  end;
end;

procedure TestStableSoftmax;
var
  Logits: TTensor;
  Probabilities: TTensor;
  Sum: Single;
begin
  Logits := TTensor.Create([3]);
  Probabilities := nil;
  try
    Logits.SetFlatValue(0, 1000);
    Logits.SetFlatValue(1, 1001);
    Logits.SetFlatValue(2, 1002);
    Probabilities := Softmax(Logits);
    Sum := Probabilities.FlatValue(0) +
      Probabilities.FlatValue(1) + Probabilities.FlatValue(2);
    AssertTrue(Abs(Sum - 1) < 1.0E-6,
      'As probabilidades devem somar um.');
    AssertTrue(Probabilities.FlatValue(2) > Probabilities.FlatValue(1),
      'O maior logit deve produzir a maior probabilidade.');
  finally
    Probabilities.Free;
    Logits.Free;
  end;
end;

procedure TestCrossEntropyUniform;
var
  Logits: TTensor;
  Loss: Single;
begin
  Logits := TTensor.Create([3]);
  try
    Logits.Fill(0);
    Loss := CrossEntropyFromLogits(Logits, 1);
    AssertTrue(Abs(Loss - Ln(3)) < 1.0E-6,
      'Três logits iguais devem produzir loss ln(3).');
  finally
    Logits.Free;
  end;
end;

procedure TestCrossEntropyShiftInvariant;
var
  BaseLogits: TTensor;
  ShiftedLogits: TTensor;
begin
  BaseLogits := TTensor.Create([3]);
  ShiftedLogits := TTensor.Create([3]);
  try
    BaseLogits.SetFlatValue(0, 1);
    BaseLogits.SetFlatValue(1, 2);
    BaseLogits.SetFlatValue(2, 3);
    ShiftedLogits.SetFlatValue(0, 1001);
    ShiftedLogits.SetFlatValue(1, 1002);
    ShiftedLogits.SetFlatValue(2, 1003);
    AssertTrue(Abs(CrossEntropyFromLogits(BaseLogits, 2) -
      CrossEntropyFromLogits(ShiftedLogits, 2)) < 1.0E-5,
      'Somar constante aos logits não deve alterar a loss.');
  finally
    ShiftedLogits.Free;
    BaseLogits.Free;
  end;
end;

function HalfSquaredLoss(const AOutput: TTensor): Single;
var
  Value: Single;
begin
  Value := AOutput.FlatValue(0);
  Result := 0.5 * Value * Value;
end;

procedure TestLinearGradientCheck;
const
  Epsilon: Single = 1.0E-3;
var
  AnalyticGradient: Single;
  GradInput: TTensor;
  GradOutput: TTensor;
  Input: TTensor;
  Layer: TLinearLayer;
  LossMinus: Single;
  LossPlus: Single;
  NumericalGradient: Single;
  OriginalWeight: Single;
  Output: TTensor;
  Random: TXorShift64Star;
begin
  Random.Initialize(7);
  Layer := TLinearLayer.Create(2, 1, Random);
  Input := TTensor.Create([2]);
  GradOutput := TTensor.Create([1]);
  Output := nil;
  GradInput := nil;
  try
    Layer.SetWeight(0, 0, 0.3);
    Layer.SetWeight(0, 1, -0.2);
    Layer.SetBias(0, 0.1);
    Input.SetFlatValue(0, 0.4);
    Input.SetFlatValue(1, -0.7);

    Output := Layer.Forward(Input);
    GradOutput.SetFlatValue(0, Output.FlatValue(0));
    Layer.ZeroGrad;
    GradInput := Layer.Backward(GradOutput);
    AnalyticGradient := Layer.WeightGradientAt(0, 1);
    FreeAndNil(Output);

    OriginalWeight := Layer.WeightAt(0, 1);
    Layer.SetWeight(0, 1, OriginalWeight + Epsilon);
    Output := Layer.Forward(Input);
    LossPlus := HalfSquaredLoss(Output);
    FreeAndNil(Output);

    Layer.SetWeight(0, 1, OriginalWeight - Epsilon);
    Output := Layer.Forward(Input);
    LossMinus := HalfSquaredLoss(Output);
    NumericalGradient := (LossPlus - LossMinus) / (2 * Epsilon);
    Layer.SetWeight(0, 1, OriginalWeight);

    AssertTrue(Abs(AnalyticGradient - NumericalGradient) < 1.0E-4,
      'Gradiente analítico deve coincidir com diferenças finitas.');
    AssertTrue(Abs(GradInput.FlatValue(0) -
      0.3 * GradOutput.FlatValue(0)) < 1.0E-6,
      'Backward deve devolver o gradiente da entrada.');
  finally
    Output.Free;
    GradInput.Free;
    GradOutput.Free;
    Input.Free;
    Layer.Free;
  end;
end;

procedure TestGradientClipScale;
var
  GradInput: TTensor;
  GradOutput: TTensor;
  Input: TTensor;
  Layer: TLinearLayer;
  Output: TTensor;
  Random: TXorShift64Star;
  Scale: Single;
begin
  Random.Initialize(8);
  Layer := TLinearLayer.Create(1, 1, Random);
  Input := TTensor.Create([1]);
  GradOutput := TTensor.Create([1]);
  Output := nil;
  GradInput := nil;
  try
    Input.SetFlatValue(0, 3);
    GradOutput.SetFlatValue(0, 4);
    Output := Layer.Forward(Input);
    Layer.ZeroGrad;
    GradInput := Layer.Backward(GradOutput);
    Scale := LinearGradientClipScale(Layer, 2);
    AssertTrue(Abs(LinearGradientGlobalNorm(Layer) * Scale - 2) < 1.0E-5,
      'Clipping deve reduzir a norma efetiva para dois.');
  finally
    GradInput.Free;
    Output.Free;
    GradOutput.Free;
    Input.Free;
    Layer.Free;
  end;
end;

procedure TestAdamLearnsTinyExample;
var
  ErrorValue: Single;
  GradInput: TTensor;
  GradOutput: TTensor;
  Input: TTensor;
  Layer: TLinearLayer;
  Optimizer: TAdamOptimizer;
  Output: TTensor;
  Random: TXorShift64Star;
  Step: Integer;
begin
  Random.Initialize(9);
  Layer := TLinearLayer.Create(1, 1, Random);
  Optimizer := TAdamOptimizer.Create(Layer, 0.05);
  Input := TTensor.Create([1]);
  GradOutput := TTensor.Create([1]);
  Output := nil;
  GradInput := nil;
  try
    Layer.SetWeight(0, 0, 0);
    Layer.SetBias(0, 0);
    Input.SetFlatValue(0, 1);
    for Step := 1 to 200 do
    begin
      FreeAndNil(Output);
      Output := Layer.Forward(Input);
      ErrorValue := Output.FlatValue(0) - 2;
      GradOutput.SetFlatValue(0, ErrorValue);
      Layer.ZeroGrad;
      FreeAndNil(GradInput);
      GradInput := Layer.Backward(GradOutput);
      Optimizer.Step(1);
    end;
    FreeAndNil(Output);
    Output := Layer.Forward(Input);
    ErrorValue := Output.FlatValue(0) - 2;
    AssertTrue(0.5 * ErrorValue * ErrorValue < 1.0E-6,
      'Adam deve ajustar o exemplo minúsculo até loss inferior a 1e-6.');
  finally
    GradInput.Free;
    Output.Free;
    GradOutput.Free;
    Input.Free;
    Optimizer.Free;
    Layer.Free;
  end;
end;

procedure TestTokenPositionEmbeddingForward;
var
  Embedding: TTokenPositionEmbedding;
  Output: TTensor;
  Random: TXorShift64Star;
  Tokens: TArray<Integer>;
begin
  Random.Initialize(10);
  Embedding := TTokenPositionEmbedding.Create(3, 4, 2, Random);
  Output := nil;
  try
    Embedding.TokenTable.SetValue(1, 0, 1.0);
    Embedding.TokenTable.SetValue(1, 1, 2.0);
    Embedding.PositionTable.SetValue(0, 0, 0.1);
    Embedding.PositionTable.SetValue(0, 1, 0.2);
    Tokens := TArray<Integer>.Create(1);
    Output := Embedding.Forward(Tokens);
    AssertTrue(Abs(Output.ValueAt([0, 0]) - 1.1) < 1.0E-6,
      'Embedding deve somar token e posição no componente zero.');
    AssertTrue(Abs(Output.ValueAt([0, 1]) - 2.2) < 1.0E-6,
      'Embedding deve somar token e posição no componente um.');
  finally
    Output.Free;
    Embedding.Free;
  end;
end;

procedure TestEmbeddingBackwardAccumulatesRepeatedToken;
var
  Embedding: TTokenPositionEmbedding;
  GradOutput: TTensor;
  Output: TTensor;
  Random: TXorShift64Star;
  Tokens: TArray<Integer>;
begin
  Random.Initialize(11);
  Embedding := TTokenPositionEmbedding.Create(3, 4, 2, Random);
  GradOutput := TTensor.Create([2, 2]);
  Output := nil;
  try
    Tokens := TArray<Integer>.Create(1, 1);
    Output := Embedding.Forward(Tokens);
    GradOutput.Fill(1);
    Embedding.ZeroGrad;
    Embedding.Backward(GradOutput);
    AssertTrue(Abs(Embedding.TokenTable.GradientAt(1, 0) - 2) < 1.0E-6,
      'Token repetido deve acumular duas contribuições.');
    AssertTrue(Abs(Embedding.PositionTable.GradientAt(0, 0) - 1) < 1.0E-6,
      'Cada posição deve receber sua própria contribuição.');
  finally
    Output.Free;
    GradOutput.Free;
    Embedding.Free;
  end;
end;

procedure TestCausalAttentionUniformWeights;
var
  Keys: TTensor;
  Output: TTensor;
  Queries: TTensor;
  Values: TTensor;
  Weights: TTensor;
begin
  Queries := TTensor.Create([3, 1]);
  Keys := TTensor.Create([3, 1]);
  Values := TTensor.Create([3, 1]);
  Output := nil;
  Weights := nil;
  try
    Queries.Fill(0);
    Keys.Fill(0);
    Values.SetValue(10, [0, 0]);
    Values.SetValue(20, [1, 0]);
    Values.SetValue(30, [2, 0]);
    Output := CausalScaledDotProductAttention(
      Queries, Keys, Values, Weights);
    AssertTrue(Abs(Weights.ValueAt([0, 0]) - 1) < 1.0E-6,
      'A primeira posição deve olhar somente para si.');
    AssertTrue(Abs(Weights.ValueAt([1, 0]) - 0.5) < 1.0E-6,
      'A segunda posição deve dividir pesos entre duas chaves.');
    AssertTrue(Weights.ValueAt([1, 2]) = 0,
      'A máscara causal deve zerar a posição futura.');
    AssertTrue(Abs(Output.ValueAt([2, 0]) - 20) < 1.0E-6,
      'A terceira saída deve ser a média 10, 20 e 30.');
  finally
    Weights.Free;
    Output.Free;
    Values.Free;
    Keys.Free;
    Queries.Free;
  end;
end;

procedure TestCausalAttentionIgnoresFutureValue;
var
  FirstOutput: TTensor;
  FirstWeights: TTensor;
  Keys: TTensor;
  Queries: TTensor;
  SecondOutput: TTensor;
  SecondWeights: TTensor;
  Values: TTensor;
begin
  Queries := TTensor.Create([3, 1]);
  Keys := TTensor.Create([3, 1]);
  Values := TTensor.Create([3, 1]);
  FirstOutput := nil;
  FirstWeights := nil;
  SecondOutput := nil;
  SecondWeights := nil;
  try
    Queries.Fill(0);
    Keys.Fill(0);
    Values.SetValue(10, [0, 0]);
    Values.SetValue(20, [1, 0]);
    Values.SetValue(30, [2, 0]);
    FirstOutput := CausalScaledDotProductAttention(
      Queries, Keys, Values, FirstWeights);
    Values.SetValue(999, [2, 0]);
    SecondOutput := CausalScaledDotProductAttention(
      Queries, Keys, Values, SecondWeights);
    AssertTrue(FirstOutput.ValueAt([0, 0]) =
      SecondOutput.ValueAt([0, 0]),
      'Mudar o futuro não pode alterar a posição zero.');
    AssertTrue(FirstOutput.ValueAt([1, 0]) =
      SecondOutput.ValueAt([1, 0]),
      'Mudar o futuro não pode alterar a posição um.');
  finally
    SecondWeights.Free;
    SecondOutput.Free;
    FirstWeights.Free;
    FirstOutput.Free;
    Values.Free;
    Keys.Free;
    Queries.Free;
  end;
end;

procedure TestMultiHeadAttentionSplitsAndRecombines;
var
  Attention: TMultiHeadCausalSelfAttention;
  Column: Integer;
  Input: TTensor;
  Output: TTensor;
  Random: TXorShift64Star;
  Row: Integer;
begin
  Random.Initialize(12);
  Attention := TMultiHeadCausalSelfAttention.Create(2, 2, Random);
  Input := TTensor.Create([3, 2]);
  Output := nil;
  try
    for Row := 0 to 1 do
      for Column := 0 to 1 do
      begin
        Attention.SetQueryWeight(Row, Column, 0);
        Attention.SetKeyWeight(Row, Column, 0);
        Attention.SetValueWeight(Row, Column, 0);
        Attention.SetOutputWeight(Row, Column, 0);
      end;
    Attention.SetValueWeight(0, 0, 1);
    Attention.SetValueWeight(1, 1, 1);
    Attention.SetOutputWeight(0, 0, 1);
    Attention.SetOutputWeight(1, 1, 1);
    Input.SetValue(10, [0, 0]);
    Input.SetValue(100, [0, 1]);
    Input.SetValue(20, [1, 0]);
    Input.SetValue(200, [1, 1]);
    Input.SetValue(30, [2, 0]);
    Input.SetValue(300, [2, 1]);
    Output := Attention.Forward(Input);
    AssertTrue(Abs(Output.ValueAt([1, 0]) - 15) < 1.0E-6,
      'A primeira cabeça deve calcular média 15.');
    AssertTrue(Abs(Output.ValueAt([1, 1]) - 150) < 1.0E-6,
      'A segunda cabeça deve calcular média 150.');
    AssertTrue(Abs(Output.ValueAt([2, 1]) - 200) < 1.0E-6,
      'A projeção deve recombinar a segunda cabeça.');
  finally
    Output.Free;
    Input.Free;
    Attention.Free;
  end;
end;

procedure TestMultiHeadAttentionRejectsInvalidDivision;
var
  Attention: TMultiHeadCausalSelfAttention;
  Random: TXorShift64Star;
  RaisedExpectedError: Boolean;
begin
  Random.Initialize(13);
  Attention := nil;
  RaisedExpectedError := False;
  try
    try
      Attention := TMultiHeadCausalSelfAttention.Create(5, 2, Random);
    except
      on E: EDelphiLMTensorError do
        RaisedExpectedError := True;
    end;
    AssertTrue(RaisedExpectedError,
      'Dimensão deve ser divisível pela quantidade de cabeças.');
  finally
    Attention.Free;
  end;
end;

procedure TestLayerNormNormalizesEachPosition;
var
  Input: TTensor;
  LayerNorm: TLayerNorm;
  Mean: Single;
  MeanSquare: Single;
  Output: TTensor;
begin
  LayerNorm := TLayerNorm.Create(3);
  Input := TTensor.Create([2, 3]);
  Output := nil;
  try
    Input.SetValue(1, [0, 0]);
    Input.SetValue(2, [0, 1]);
    Input.SetValue(3, [0, 2]);
    Input.SetValue(10, [1, 0]);
    Input.SetValue(10, [1, 1]);
    Input.SetValue(10, [1, 2]);
    Output := LayerNorm.Forward(Input);
    Mean := (Output.ValueAt([0, 0]) + Output.ValueAt([0, 1]) +
      Output.ValueAt([0, 2])) / 3;
    MeanSquare := (Sqr(Output.ValueAt([0, 0])) +
      Sqr(Output.ValueAt([0, 1])) + Sqr(Output.ValueAt([0, 2]))) / 3;
    AssertTrue(Abs(Mean) < 1.0E-6,
      'A média da posição normalizada deve ser zero.');
    AssertTrue(Abs(MeanSquare - 1) < 1.0E-4,
      'A variância da posição não constante deve ficar próxima de um.');
    AssertTrue(Abs(Output.ValueAt([1, 0])) < 1.0E-6,
      'Uma posição constante deve ser transformada em zeros.');
  finally
    Output.Free;
    Input.Free;
    LayerNorm.Free;
  end;
end;

procedure TestTransformerBlockResidualIdentity;
var
  Block: TTransformerBlock;
  Column: Integer;
  Input: TTensor;
  Output: TTensor;
  Random: TXorShift64Star;
  Row: Integer;
begin
  Random.Initialize(14);
  Block := TTransformerBlock.Create(2, 2, 4, Random);
  Input := TTensor.Create([2, 2]);
  Output := nil;
  try
    for Row := 0 to 1 do
      for Column := 0 to 1 do
        Block.Attention.SetOutputWeight(Row, Column, 0);
    for Row := 0 to 1 do
      for Column := 0 to 3 do
        Block.SetFeedForwardOutputWeight(Row, Column, 0);
    Input.SetValue(1, [0, 0]);
    Input.SetValue(2, [0, 1]);
    Input.SetValue(3, [1, 0]);
    Input.SetValue(4, [1, 1]);
    Output := Block.Forward(Input);
    for Row := 0 to 1 do
      for Column := 0 to 1 do
        AssertTrue(Output.ValueAt([Row, Column]) =
          Input.ValueAt([Row, Column]),
          'Ramos zerados devem preservar a entrada pelo residual.');
  finally
    Output.Free;
    Input.Free;
    Block.Free;
  end;
end;

procedure ZeroTransformerBranches(const ABlock: TTransformerBlock;
  const AEmbeddingDimension, AFeedForwardDimension: Integer);
var
  Column: Integer;
  Row: Integer;
begin
  for Row := 0 to AEmbeddingDimension - 1 do
    for Column := 0 to AEmbeddingDimension - 1 do
      ABlock.Attention.SetOutputWeight(Row, Column, 0);
  for Row := 0 to AEmbeddingDimension - 1 do
    for Column := 0 to AFeedForwardDimension - 1 do
      ABlock.SetFeedForwardOutputWeight(Row, Column, 0);
end;

procedure TestLanguageModelProducesLogitsWithTiedWeights;
var
  Config: TDelphiLMConfig;
  Logits: TTensor;
  Model: TDelphiLanguageModel;
  Tokens: TArray<Integer>;
begin
  Config := TDelphiLMConfig.Default;
  Config.VocabularySize := 3;
  Config.ContextLength := 2;
  Config.EmbeddingDimension := 2;
  Config.BlockCount := 1;
  Config.AttentionHeadCount := 2;
  Config.FeedForwardDimension := 4;
  Model := TDelphiLanguageModel.Create(Config);
  Logits := nil;
  try
    ZeroTransformerBranches(Model.BlockAt(0), 2, 4);
    Model.Embeddings.TokenTable.SetValue(0, 0, 1);
    Model.Embeddings.TokenTable.SetValue(0, 1, 3);
    Model.Embeddings.TokenTable.SetValue(1, 0, -1);
    Model.Embeddings.TokenTable.SetValue(1, 1, 1);
    Model.Embeddings.TokenTable.SetValue(2, 0, 1);
    Model.Embeddings.TokenTable.SetValue(2, 1, -1);
    Model.Embeddings.PositionTable.SetValue(0, 0, 0);
    Model.Embeddings.PositionTable.SetValue(0, 1, 0);
    Tokens := TArray<Integer>.Create(0);
    Logits := Model.Forward(Tokens);
    AssertTrue((Logits.Dimension(0) = 1) and
      (Logits.Dimension(1) = 3),
      'O modelo deve produzir um logit por token do vocabulário.');
    AssertTrue(Logits.ValueAt([0, 1]) > 1.9,
      'O token 1 deve receber logit positivo pelos pesos compartilhados.');
    AssertTrue(Logits.ValueAt([0, 2]) < -1.9,
      'O token 2 deve receber logit negativo pelos pesos compartilhados.');
  finally
    Logits.Free;
    Model.Free;
  end;
end;

procedure TestLanguageModelRejectsLongContext;
var
  Config: TDelphiLMConfig;
  Logits: TTensor;
  Model: TDelphiLanguageModel;
  RaisedExpectedError: Boolean;
  Tokens: TArray<Integer>;
begin
  Config := TDelphiLMConfig.Default;
  Config.VocabularySize := 3;
  Config.ContextLength := 2;
  Config.EmbeddingDimension := 2;
  Config.BlockCount := 1;
  Config.AttentionHeadCount := 1;
  Config.FeedForwardDimension := 4;
  Model := TDelphiLanguageModel.Create(Config);
  Logits := nil;
  RaisedExpectedError := False;
  try
    Tokens := TArray<Integer>.Create(0, 1, 2);
    try
      Logits := Model.Forward(Tokens);
    except
      on E: EDelphiLMTensorError do
        RaisedExpectedError := True;
    end;
    AssertTrue(RaisedExpectedError,
      'O modelo deve rejeitar sequência maior que o contexto máximo.');
  finally
    Logits.Free;
    Model.Free;
  end;
end;

procedure TestFullModelLearnsTinySequence;
var
  Config: TDelphiLMConfig;
  FinalLoss: Single;
  InitialLoss: Single;
  InputBatch: TTokenBatch;
  Inputs: TArray<Integer>;
  Model: TDelphiLanguageModel;
  Optimizer: TModelAdamOptimizer;
  Step: Integer;
  Targets: TArray<Integer>;
  TargetBatch: TTokenBatch;
begin
  Config := TDelphiLMConfig.Default;
  Config.VocabularySize := 3;
  Config.ContextLength := 2;
  Config.EmbeddingDimension := 2;
  Config.BlockCount := 1;
  Config.AttentionHeadCount := 1;
  Config.FeedForwardDimension := 4;
  Config.Seed := 15;
  Model := TDelphiLanguageModel.Create(Config);
  Optimizer := TModelAdamOptimizer.Create(Model, 0.02);
  try
    Inputs := TArray<Integer>.Create(0, 0);
    Targets := TArray<Integer>.Create(1, 1);
    InputBatch := TTokenBatch.Create(Inputs, Copy(Inputs));
    TargetBatch := TTokenBatch.Create(Targets, Copy(Targets));
    InitialLoss := TrainBatch(Model, Optimizer, InputBatch, TargetBatch);
    for Step := 2 to 120 do
      FinalLoss := TrainBatch(Model, Optimizer, InputBatch, TargetBatch);
    AssertTrue(FinalLoss < InitialLoss * 0.1,
      'O modelo completo deve reduzir a loss do batch minúsculo em 90%.');
    AssertTrue(FinalLoss < 0.05,
      'O modelo completo deve memorizar o batch minúsculo.');
    Writeln(Format('     loss: %.6f -> %.6f', [InitialLoss, FinalLoss]));
  finally
    Optimizer.Free;
    Model.Free;
  end;
end;

procedure TestTopKOneAlwaysChoosesMaximum;
var Logits: TTensor; Random: TXorShift64Star;
begin
  Logits := TTensor.Create([1, 3]);
  try
    Logits.SetValue(-1, [0, 0]); Logits.SetValue(4, [0, 1]);
    Logits.SetValue(2, [0, 2]); Random.Initialize(16);
    AssertTrue(SampleNextToken(Logits, 0, 1, 1, Random) = 1,
      'Top-K igual a um deve escolher o maior logit.');
  finally Logits.Free end;
end;

procedure TestCheckpointRestoresExactLogits;
var A, B: TDelphiLanguageModel; C: TDelphiLMConfig; FileName: string;
  I: Integer; LA, LB: TTensor; Tokens: TArray<Integer>;
begin
  C := TDelphiLMConfig.Default; C.VocabularySize := 3; C.ContextLength := 2;
  C.EmbeddingDimension := 2; C.BlockCount := 1; C.AttentionHeadCount := 1;
  C.FeedForwardDimension := 4; C.Seed := 17;
  A := TDelphiLanguageModel.Create(C); C.Seed := 999;
  B := TDelphiLanguageModel.Create(C); LA := nil; LB := nil;
  FileName := TPath.GetTempFileName;
  try
    SaveCheckpoint(A, FileName); LoadCheckpoint(B, FileName);
    Tokens := TArray<Integer>.Create(0, 1); LA := A.Forward(Tokens); LB := B.Forward(Tokens);
    for I := 0 to LA.ElementCount - 1 do AssertTrue(LA.FlatValue(I) = LB.FlatValue(I),
      'Checkpoint deve restaurar logits bit a bit.');
  finally
    LB.Free; LA.Free; B.Free; A.Free;
    if TFile.Exists(FileName) then TFile.Delete(FileName);
  end;
end;

procedure RunTest(const AName: string; const ATest: TProc);
begin
  ATest();
  Writeln('[OK] ', AName);
end;

begin
  try
    RunTest('configuração-base', TestDefaultConfig);
    RunTest('validação de heads', TestInvalidHeadCount);
    RunTest('aleatoriedade determinística', TestDeterministicRandom);
    RunTest('tokenizer round-trip', TestTokenizerRoundTrip);
    RunTest('normalização NFC', TestTokenizerNormalizesNFC);
    RunTest('tensor row-major', TestTensorRowMajor);
    RunTest('produto escalar', TestDotProduct);
    RunTest('multiplicação matricial', TestMatrixMultiply);
    RunTest('camada linear forward', TestLinearForward);
    RunTest('ativações', TestActivations);
    RunTest('inicialização linear determinística',
      TestLinearInitializationIsDeterministic);
    RunTest('softmax estável', TestStableSoftmax);
    RunTest('cross-entropy uniforme', TestCrossEntropyUniform);
    RunTest('cross-entropy invariante a deslocamento',
      TestCrossEntropyShiftInvariant);
    RunTest('gradient check da camada linear', TestLinearGradientCheck);
    RunTest('clipping por norma global', TestGradientClipScale);
    RunTest('Adam aprende exemplo minúsculo', TestAdamLearnsTinyExample);
    RunTest('embedding de token e posição', TestTokenPositionEmbeddingForward);
    RunTest('backward de embedding acumula repetição',
      TestEmbeddingBackwardAccumulatesRepeatedToken);
    RunTest('atenção causal com pesos uniformes',
      TestCausalAttentionUniformWeights);
    RunTest('atenção causal ignora valor futuro',
      TestCausalAttentionIgnoresFutureValue);
    RunTest('multi-head divide e recombina',
      TestMultiHeadAttentionSplitsAndRecombines);
    RunTest('multi-head rejeita divisão inválida',
      TestMultiHeadAttentionRejectsInvalidDivision);
    RunTest('LayerNorm por posição', TestLayerNormNormalizesEachPosition);
    RunTest('bloco Transformer preserva residual',
      TestTransformerBlockResidualIdentity);
    RunTest('modelo produz logits com pesos compartilhados',
      TestLanguageModelProducesLogitsWithTiedWeights);
    RunTest('modelo rejeita contexto longo',
      TestLanguageModelRejectsLongContext);
    RunTest('modelo completo aprende sequência minúscula',
      TestFullModelLearnsTinySequence);
    RunTest('Top-K um escolhe o máximo', TestTopKOneAlwaysChoosesMaximum);
    RunTest('checkpoint restaura logits exatos', TestCheckpointRestoresExactLogits);
    Writeln('30 testes aprovados.');
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, '[FALHOU] ', E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
