{ Licensed under the Apache License, Version 2.0. See LICENSE. }
program DelphiLM_Tests;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  DelphiLM.Core.Config in '..\src\DelphiLM.Core.Config.pas',
  DelphiLM.Core.Random in '..\src\DelphiLM.Core.Random.pas',
  DelphiLM.Data.Tokenizer in '..\src\DelphiLM.Data.Tokenizer.pas',
  DelphiLM.Math.Tensor in '..\src\DelphiLM.Math.Tensor.pas';

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
    Writeln('8 testes aprovados.');
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, '[FALHOU] ', E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
