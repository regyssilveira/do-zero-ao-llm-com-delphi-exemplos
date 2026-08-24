{ Licensed under the Apache License, Version 2.0. See LICENSE. }
program DelphiLM_Benchmark;

{$APPTYPE CONSOLE}

uses
  System.Diagnostics,
  System.IOUtils,
  System.SysUtils,
  Winapi.PsAPI,
  Winapi.Windows,
  DelphiLM.Core.Config in '..\src\DelphiLM.Core.Config.pas',
  DelphiLM.Core.Random in '..\src\DelphiLM.Core.Random.pas',
  DelphiLM.Data.Tokenizer in '..\src\DelphiLM.Data.Tokenizer.pas',
  DelphiLM.Math.Tensor in '..\src\DelphiLM.Math.Tensor.pas',
  DelphiLM.Neural.Parameters in '..\src\DelphiLM.Neural.Parameters.pas',
  DelphiLM.Neural.Layers in '..\src\DelphiLM.Neural.Layers.pas',
  DelphiLM.Neural.Loss in '..\src\DelphiLM.Neural.Loss.pas',
  DelphiLM.Neural.Optimizers in '..\src\DelphiLM.Neural.Optimizers.pas',
  DelphiLM.Transformer.Embeddings in '..\src\DelphiLM.Transformer.Embeddings.pas',
  DelphiLM.Transformer.Attention in '..\src\DelphiLM.Transformer.Attention.pas',
  DelphiLM.Transformer.MultiHeadAttention in '..\src\DelphiLM.Transformer.MultiHeadAttention.pas',
  DelphiLM.Transformer.Block in '..\src\DelphiLM.Transformer.Block.pas',
  DelphiLM.Model.LanguageModel in '..\src\DelphiLM.Model.LanguageModel.pas',
  DelphiLM.Model.Generation in '..\src\DelphiLM.Model.Generation.pas',
  DelphiLM.Model.Training in '..\src\DelphiLM.Model.Training.pas';

function CorpusPath: string;
begin
  if ParamCount >= 1 then
    Exit(TPath.GetFullPath(ParamStr(1)));
  Result := TPath.GetFullPath(TPath.Combine(ExtractFilePath(ParamStr(0)),
    '..\..\..\..\datasets\corpus-base.txt'));
end;

function RequestedSteps: Integer;
begin
  Result := 20;
  if ParamCount >= 2 then
    Result := StrToInt(ParamStr(2));
  if Result <= 0 then
    raise EArgumentOutOfRangeException.Create('Steps deve ser positivo.');
end;

procedure MakeBatch(const ATokens: TArray<Integer>; const AStart,
  ACount: Integer; const AConfig: TDelphiLMConfig;
  var ARandom: TXorShift64Star; out AInputs, ATargets: TTokenBatch);
var
  Example: Integer;
  Offset: Integer;
  Position: Integer;
  WindowCount: Integer;
begin
  WindowCount := ACount - AConfig.ContextLength;
  if WindowCount <= 0 then
    raise EArgumentException.Create('Segmento menor que a janela de contexto.');
  SetLength(AInputs, AConfig.BatchSize);
  SetLength(ATargets, AConfig.BatchSize);
  for Example := 0 to AConfig.BatchSize - 1 do
  begin
    Offset := AStart + Integer(ARandom.NextUInt64 mod UInt64(WindowCount));
    SetLength(AInputs[Example], AConfig.ContextLength);
    SetLength(ATargets[Example], AConfig.ContextLength);
    for Position := 0 to AConfig.ContextLength - 1 do
    begin
      AInputs[Example][Position] := ATokens[Offset + Position];
      ATargets[Example][Position] := ATokens[Offset + Position + 1];
    end;
  end;
end;

function BatchLoss(const AModel: TDelphiLanguageModel;
  const AInputs, ATargets: TTokenBatch): Single;
var
  Example: Integer;
  GradLogits: TTensor;
  Logits: TTensor;
begin
  Result := 0;
  for Example := 0 to High(AInputs) do
  begin
    Logits := AModel.Forward(AInputs[Example]);
    GradLogits := nil;
    try
      Result := Result + NextTokenLossAndGradient(
        Logits, ATargets[Example], GradLogits) / Length(AInputs);
    finally
      GradLogits.Free;
      Logits.Free;
    end;
  end;
end;

function PeakWorkingSetBytes: UInt64;
var
  Counters: TProcessMemoryCounters;
begin
  Counters.cb := SizeOf(Counters);
  if not GetProcessMemoryInfo(GetCurrentProcess, @Counters, SizeOf(Counters)) then
    RaiseLastOSError;
  Result := Counters.PeakWorkingSetSize;
end;

var
  Config: TDelphiLMConfig;
  Corpus: string;
  DataPath: string;
  ElapsedSeconds: Double;
  FinalTrainLoss: Single;
  FinalValidationLoss: Single;
  FormatSettings: TFormatSettings;
  Generated: TArray<Integer>;
  GenerationRandom: TXorShift64Star;
  GenerationSeconds: Double;
  GenerationStopwatch: TStopwatch;
  InitialTrainLoss: Single;
  InitialValidationLoss: Single;
  Inputs: TTokenBatch;
  Model: TDelphiLanguageModel;
  Optimizer: TModelAdamOptimizer;
  Prompt: TArray<Integer>;
  Random: TXorShift64Star;
  Split: Integer;
  Step: Integer;
  Steps: Integer;
  Stopwatch: TStopwatch;
  Targets: TTokenBatch;
  Tokenizer: TCharacterTokenizer;
  Tokens: TArray<Integer>;
  TrainTokenCount: Int64;
  ValidationInputs: TTokenBatch;
  ValidationRandom: TXorShift64Star;
  ValidationTargets: TTokenBatch;
begin
  try
    FormatSettings := TFormatSettings.Invariant;
    DataPath := CorpusPath;
    Steps := RequestedSteps;
    Corpus := TFile.ReadAllText(DataPath, TEncoding.UTF8);
    Tokenizer := TCharacterTokenizer.Create(Corpus);
    try
      Tokens := Tokenizer.Encode(Corpus);
      Split := (Length(Tokens) * 9) div 10;
      Config := TDelphiLMConfig.Default;
      Config.VocabularySize := Tokenizer.VocabularySize;
      Config.Validate;
      Random.Initialize(Config.Seed);
      ValidationRandom.Initialize(Config.Seed + 1);
      MakeBatch(Tokens, 0, Split, Config, Random, Inputs, Targets);
      MakeBatch(Tokens, Split, Length(Tokens) - Split, Config,
        ValidationRandom, ValidationInputs, ValidationTargets);
      Model := TDelphiLanguageModel.Create(Config);
      Optimizer := TModelAdamOptimizer.Create(Model, Config.LearningRate);
      try
        InitialTrainLoss := BatchLoss(Model, Inputs, Targets);
        InitialValidationLoss := BatchLoss(
          Model, ValidationInputs, ValidationTargets);
        Stopwatch := TStopwatch.StartNew;
        FinalTrainLoss := InitialTrainLoss;
        for Step := 1 to Steps do
        begin
          MakeBatch(Tokens, 0, Split, Config, Random, Inputs, Targets);
          FinalTrainLoss := TrainBatch(Model, Optimizer, Inputs, Targets);
        end;
        Stopwatch.Stop;
        FinalValidationLoss := BatchLoss(
          Model, ValidationInputs, ValidationTargets);
        Prompt := Copy(ValidationInputs[0], 0, 16);
        GenerationRandom.Initialize(Config.Seed + 2);
        GenerationStopwatch := TStopwatch.StartNew;
        Generated := GenerateTokens(Model, Prompt, 64, 0.8, 8,
          GenerationRandom);
        GenerationStopwatch.Stop;
        GenerationSeconds := GenerationStopwatch.Elapsed.TotalSeconds;
        ElapsedSeconds := Stopwatch.Elapsed.TotalSeconds;
        TrainTokenCount := Int64(Steps) * Config.BatchSize *
          Config.ContextLength;
        Writeln('status=ok');
        Writeln('corpus=', DataPath);
        Writeln('corpus_utf16_units=', Length(Tokens));
        Writeln('vocabulary_size=', Config.VocabularySize);
        Writeln('seed=', Config.Seed);
        Writeln('steps=', Steps);
        Writeln('batch_size=', Config.BatchSize);
        Writeln('context_length=', Config.ContextLength);
        Writeln('parameters=', Config.ParameterCount);
        Writeln('train_tokens=', TrainTokenCount);
        Writeln('train_loss_initial=', FloatToStr(InitialTrainLoss, FormatSettings));
        Writeln('train_loss_final=', FloatToStr(FinalTrainLoss, FormatSettings));
        Writeln('validation_loss_initial=', FloatToStr(
          InitialValidationLoss, FormatSettings));
        Writeln('validation_loss_final=', FloatToStr(
          FinalValidationLoss, FormatSettings));
        Writeln('elapsed_seconds=', FloatToStr(ElapsedSeconds, FormatSettings));
        Writeln('tokens_per_second=', FloatToStr(
          TrainTokenCount / ElapsedSeconds, FormatSettings));
        Writeln('generation_new_tokens=', Length(Generated) - Length(Prompt));
        Writeln('generation_seconds=', FloatToStr(
          GenerationSeconds, FormatSettings));
        Writeln('generation_tokens_per_second=', FloatToStr(
          (Length(Generated) - Length(Prompt)) / GenerationSeconds,
          FormatSettings));
        Writeln('generation_first_token_seconds=', FloatToStr(
          GenerationSeconds / (Length(Generated) - Length(Prompt)),
          FormatSettings));
        Writeln('peak_working_set_bytes=', PeakWorkingSetBytes);
      finally
        Optimizer.Free;
        Model.Free;
      end;
    finally
      Tokenizer.Free;
    end;
  except
    on E: Exception do
    begin
      Writeln(ErrOutput, 'status=failed');
      Writeln(ErrOutput, 'error=', E.ClassName, ': ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
