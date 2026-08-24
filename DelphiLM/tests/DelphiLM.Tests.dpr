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
  DelphiLM.Neural.Loss in '..\src\DelphiLM.Neural.Loss.pas';

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
    Writeln('14 testes aprovados.');
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, '[FALHOU] ', E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
