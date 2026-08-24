{ Licensed under the Apache License, Version 2.0. See LICENSE. }
program DelphiLM_Tests;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  DelphiLM.Core.Config in '..\src\DelphiLM.Core.Config.pas',
  DelphiLM.Core.Random in '..\src\DelphiLM.Core.Random.pas',
  DelphiLM.Data.Tokenizer in '..\src\DelphiLM.Data.Tokenizer.pas';

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
    Writeln('5 testes aprovados.');
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, '[FALHOU] ', E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
